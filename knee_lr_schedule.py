import warnings
from typing import Union, List

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class KneeLRScheduler(_LRScheduler):
    """
    `Wide-minima Density Hypothesis and the Explore-Exploit Learning Rate Schedule`
    Implementation of the paper https://arxiv.org/pdf/2003.03977.pdf
    Taken from https://github.com/nikhil-iyer-97/wide-minima-density-hypothesis

    Allows the optimizer to explore the loss landscape with a high learning rate for 
    accessing and landing in a wide minimum based on the hypothesis on minima density.
    The LR is constant for the entire explore phase and the exploit phase decays the LR
    linearly to zero. Warmup is complementary and can be added to the schedule.  


    You must either provide a value for total_steps or provide a value for both
    epochs and steps_per_epoch. Warmup and explore are set to zero by default. If
    no values have been provided, it will linearly decay from max_lr to zero.

    Args:
        optimizer: Optimizer needed for training the model ( SGD/Adam ).
        max_lr (float): The peak learning required for explore phase to escape narrow minimums.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is not provided here, then it must be inferred by providing
            a value for epochs and steps_per_epoch.
        epochs (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle
            if a value for total_steps is not provided.
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the
            cycle if a value for total_steps is not provided.
        warmup (float): The percentage of the total steps spent increasing the learning rate ( default: 0 )
        explore (float): The percentage of the total steps spent on explore phase (keeping max_lr 
            as the learning rate). Default : 0
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
        verbose (bool): If ``True``, prints a message to stdout for
            each update.
    """

    def __init__(self, optimizer: Optimizer, max_lr: Union[List[float], float], total_steps: int = None,
                 epochs: int = None, steps_per_epoch: int = None, warmup: int = 0.3, explore: int = 0.4,
                 cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
                 div_factor: float = 25., final_div_factor: float = 1e4, last_epoch: int = -1, verbose: bool = False):

        # Validate optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer

        # Validate total_steps
        if total_steps is None and epochs is None and steps_per_epoch is None:
            raise ValueError("You must define either total_steps OR (epochs AND steps_per_epoch)")
        elif total_steps is not None:
            if total_steps <= 0 or not isinstance(total_steps, int):
                raise ValueError("Expected positive integer total_steps, but got {}".format(total_steps))
            self.total_steps = total_steps
        else:
            if epochs <= 0 or not isinstance(epochs, int):
                raise ValueError("Expected positive integer epochs, but got {}".format(epochs))
            if steps_per_epoch <= 0 or not isinstance(steps_per_epoch, int):
                raise ValueError("Expected positive integer steps_per_epoch, but got {}".format(steps_per_epoch))
            self.total_steps = epochs * steps_per_epoch

        # Validate warmup
        if warmup < 0 or warmup > 1 or not isinstance(warmup, float):
            raise ValueError(f"Expected float between 0 and 1 warmup, but got {warmup}")

        # Validate explore
        if explore < 0 or explore > 1 - warmup or not isinstance(explore, float):
            raise ValueError(f"Expected float between 0 and (1-warmup) explore, but got {explore}")

        self.max_lr = max_lr
        self.warmup_steps = self.total_steps * warmup
        self.explore_steps = self.total_steps * explore
        self.decay_steps = self.total_steps - (self.explore_steps + self.warmup_steps)
        self.last_epoch = last_epoch

        assert self.decay_steps >= 0

        max_lrs = self._format_param('max_lr', self.optimizer, max_lr)
        if last_epoch == -1:
            for idx, group in enumerate(self.optimizer.param_groups):
                group['initial_lr'] = max_lrs[idx] / div_factor
                group['max_lr'] = max_lrs[idx]
                group['min_lr'] = group['initial_lr'] / final_div_factor

        # Initialize momentum variables
        self.cycle_momentum = cycle_momentum
        if self.cycle_momentum:
            if 'momentum' not in self.optimizer.defaults and 'betas' not in self.optimizer.defaults:
                raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')
            self.use_beta1 = 'betas' in self.optimizer.defaults
            max_momentums = self._format_param('max_momentum', optimizer, max_momentum)
            base_momentums = self._format_param('base_momentum', optimizer, base_momentum)
            if last_epoch == -1:
                for m_momentum, b_momentum, group in zip(max_momentums, base_momentums, optimizer.param_groups):
                    if self.use_beta1:
                        _, beta2 = group['betas']
                        group['betas'] = (m_momentum, beta2)
                    else:
                        group['momentum'] = m_momentum
                    group['max_momentum'] = m_momentum
                    group['base_momentum'] = b_momentum

        super(KneeLRScheduler, self).__init__(optimizer, last_epoch, verbose)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def anneal_func(self, start, end, pct):
        """Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
        return (end - start) * pct + start

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        lrs = []
        step_num = self.last_epoch

        if step_num > self.total_steps:
            raise ValueError("Tried to step {} times. The specified number of total steps is {}"
                             .format(step_num + 1, self.total_steps))

        for group in self.optimizer.param_groups:
            if step_num <= self.warmup_steps:
                computed_lr = self.anneal_func(group['initial_lr'], group['max_lr'], step_num / self.warmup_steps)
                if self.cycle_momentum:
                    computed_momentum = self.anneal_func(group['max_momentum'], group['base_momentum'],
                                                         step_num / self.warmup_steps)
            elif step_num <= self.warmup_steps + self.explore_steps:
                computed_lr = group['max_lr']
                if self.cycle_momentum:
                    computed_momentum = group['base_momentum']
            else:
                down_step_num = step_num - (self.warmup_steps + self.explore_steps)
                computed_lr = self.anneal_func(group['max_lr'], group['min_lr'], down_step_num / self.decay_steps)
                if self.cycle_momentum:
                    computed_momentum = self.anneal_func(group['base_momentum'], group['max_momentum'],
                                                         down_step_num / self.decay_steps)

            lrs.append(computed_lr)
            if self.cycle_momentum:
                if self.use_beta1:
                    _, beta2 = group['betas']
                    group['betas'] = (computed_momentum, beta2)
                else:
                    group['momentum'] = computed_momentum
        return lrs
