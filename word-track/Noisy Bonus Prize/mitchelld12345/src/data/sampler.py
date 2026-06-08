"""Duration-bucketed batch sampler for efficient speech training."""
import random

from torch.utils.data import Sampler


class DurationBucketSampler(Sampler):

    def __init__(self, durations, max_batch_duration=480, shuffle=True, seed=42, drop_last=False):
        self.durations = durations
        self.max_batch_duration = max_batch_duration
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        sorted_indices = sorted(range(len(self.durations)), key=lambda i: self.durations[i])

        batches = []
        batch = []
        batch_dur = 0
        for idx in sorted_indices:
            dur = self.durations[idx]
            if batch and batch_dur + dur > self.max_batch_duration:
                batches.append(batch)
                batch = []
                batch_dur = 0
            batch.append(idx)
            batch_dur += dur
        if batch and not self.drop_last:
            batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        sorted_indices = sorted(range(len(self.durations)), key=lambda i: self.durations[i])
        n = 0
        batch_dur = 0
        batch_size = 0
        for idx in sorted_indices:
            dur = self.durations[idx]
            if batch_size > 0 and batch_dur + dur > self.max_batch_duration:
                n += 1
                batch_dur = 0
                batch_size = 0
            batch_dur += dur
            batch_size += 1
        if batch_size > 0 and not self.drop_last:
            n += 1
        return n

    def set_epoch(self, epoch):
        self.epoch = epoch
