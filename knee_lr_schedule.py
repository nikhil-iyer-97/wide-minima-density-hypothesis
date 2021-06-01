import torch
from torch.optim.optimizer import Optimizer

class KneeLRScheduler:

    def __init__(self, optimizer, peak_lr, warmup_steps=0, explore_steps=0, total_steps=0):
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.explore_steps = explore_steps
        self.total_steps = total_steps
        self.decay_steps = self.total_steps - (self.explore_steps + self.warmup_steps)
        self.current_step = 1

        assert self.decay_steps >= 0

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr(self.current_step)

        if not isinstance(self.optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(self.optimizer).__name__))
        
    def get_lr(self, global_step):
        if global_step <= self.warmup_steps:
            return self.peak_lr * global_step / self.warmup_steps
        elif global_step <= (self.explore_steps + self.warmup_steps):
            return self.peak_lr
        else:
            slope = -1 * self.peak_lr / self.decay_steps
            return max(0.0, self.peak_lr + slope*(global_step - (self.explore_steps + self.warmup_steps)))

    def step(self):
        self.current_step += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr(self.current_step)