import argparse
import math
import os
import random
import sys
from typing import Callable, Optional

import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer


def get_training_stats(stats):
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


class Pipeline:
    
    def __init__(self, argsv):
        
        
        parser = options.get_training_parser()
        sys.argv = argsv
        self.args = options.parse_args_and_arch(parser)
        # Setup task, e.g., translation, language modeling, etc.
        
        self.task = tasks.setup_task(self.args)

        # Load valid dataset (we load training data below, based on the latest checkpoint)
        for valid_sub_split in self.args.valid_subset.split(","):
            self.task.load_dataset(valid_sub_split, combine=False, epoch=1)

        # Build model and criterion
        self.model = self.task.build_model(self.args)
        self.criterion = self.task.build_criterion(self.args)
        
        self.trainer = Trainer(self.args, self.task, self.model, self.criterion)
        
        self.best_bleu = -1

    def validate_and_save(self, epoch_itr, valid_subsets, end_of_epoch):
        num_updates = self.trainer.get_num_updates()
        do_save = (
            self.args.save_interval_updates > 0
            and num_updates > 0
            and num_updates % self.args.save_interval_updates == 0
        ) or (end_of_epoch and epoch_itr.epoch % self.args.save_interval == 0)
        do_validate = (
            (not end_of_epoch and do_save)  # validate during mid-epoch saves
            or (end_of_epoch and epoch_itr.epoch % self.args.validate_interval == 0)
        ) and not self.args.disable_validation

        # Validate
        valid_losses = [None]
        if do_validate:
            valid_losses = self.validate(epoch_itr, valid_subsets)

        # Stopping conditions
        max_update = self.args.max_update or math.inf
        should_stop =  self.trainer.get_num_updates() >= max_update

        # Save checkpoint
        if do_save or should_stop:
            print("begin save checkpoint")
            checkpoint_utils.save_checkpoint(self.args, self.trainer, epoch_itr, valid_losses[0])

        return valid_losses, should_stop

    def train(self, epoch_itr):
        """Train the model for one epoch and return validation losses."""

        print("-"*25 + " begin training epoch {} ".format(epoch_itr.epoch) + "-"*25 + "\n")

        # Initialize data iterator
        itr = epoch_itr.next_epoch_itr(
            fix_batches_to_gpus=self.args.fix_batches_to_gpus,
            shuffle=(epoch_itr.next_epoch_idx > self.args.curriculum),
        )
        update_freq = (
            self.args.update_freq[epoch_itr.epoch - 1]
            if epoch_itr.epoch <= len(self.args.update_freq)
            else self.args.update_freq[-1]
        )
        itr = iterators.GroupedIterator(itr, update_freq)

        self.trainer.begin_epoch(epoch_itr.epoch)

        valid_subsets = self.args.valid_subset.split(",")
        should_stop = False
        for i, samples in enumerate(itr):
            with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function("train_step-%d" % i):
                log_output = self.trainer.train_step(samples)
                if log_output is None:  # OOM, overflow, ...
                    continue

            # log mid-epoch stats
            num_updates = self.trainer.get_num_updates()
            if num_updates % self.args.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                print("\rStep: {}| loss: {}| nll_loss: {} ".format(i, stats['loss'], stats['nll_loss']), end="")
                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

            end_of_epoch = not itr.has_next()
            valid_losses, should_stop = self.validate_and_save(
                epoch_itr, valid_subsets, end_of_epoch
            )
            if should_stop:
                break


        # log end-of-epoch stats
        print("end of epoch {}".format(epoch_itr.epoch))
        stats = get_training_stats(metrics.get_smoothed_values("train"))
        print("loss: {}| nll_loss: {} ".format(stats['loss'], stats['nll_loss']))
        print("-" * 73)

        # reset epoch-level meters
        metrics.reset_meters("train")
        return valid_losses, should_stop
    
    def validate(self, epoch_itr, subsets):
        """Evaluate the model on the validation set(s) and return the losses."""

        if self.args.fixed_validation_seed is not None:
            # set fixed seed for every validation
            utils.set_torch_seed(args.fixed_validation_seed)

        valid_losses = []
        for subset in subsets:
            print("\nbegin validation on \"{}\" subset".format(subset))

            # Initialize data iterator
            itr = self.trainer.get_valid_iterator(subset).next_epoch_itr(shuffle=False)

            # create a new root metrics aggregator so validation metrics
            # don't pollute other aggregators (e.g., train meters)
            with metrics.aggregate(new_root=True) as agg:
                for sample in itr:
                    self.trainer.valid_step(sample)

            # log validation stats
            stats = self.get_valid_stats(agg.get_smoothed_values())
            print("num_updates: {} | loss: {}| nll_loss: {} | bleu {} ".format(
                stats['num_updates'], stats['loss'], stats['nll_loss'], stats['bleu']))
            
            self.best_bleu = max(stats['bleu'], self.best_bleu)
            valid_losses.append(stats[self.args.best_checkpoint_metric])
        return valid_losses


    def get_valid_stats(self, stats):
        stats["num_updates"] = self.trainer.get_num_updates()
        if hasattr(checkpoint_utils.save_checkpoint, "best"):
            key = "best_{0}".format(self.args.best_checkpoint_metric)
            best_function = max if self.args.maximize_best_checkpoint_metric else min
            stats[key] = best_function(
                checkpoint_utils.save_checkpoint.best, stats[self.args.best_checkpoint_metric]
            )
        return stats
    
    def run(self):
        extra_state, epoch_itr = checkpoint_utils.load_checkpoint(self.args, self.trainer)
        lr = self.trainer.get_lr()
        train_meter = meters.StopwatchMeter()
        train_meter.start()
        max_epoch = self.args.max_epoch or math.inf


        while lr > self.args.min_lr and epoch_itr.next_epoch_idx <= max_epoch:
            # train for one epoch
            valid_losses, should_stop = self.train(epoch_itr)
            if should_stop:
                break

            # only use first validation loss to update the learning rate
            lr = self.trainer.lr_step(epoch_itr.epoch, valid_losses[0])

            epoch_itr = self.trainer.get_train_iterator(
                epoch_itr.next_epoch_idx,
                # sharded data: get train iterator for next epoch
                load_dataset=(os.pathsep in getattr(self.args, "data", "")),
            )
        train_meter.stop()

