import datetime
import errno
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist
from bisect import bisect_right


class LinearLR(torch.optim.lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small
    multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 1./3.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025    if epoch == 0
        >>> # lr = 0.03125  if epoch == 1
        >>> # lr = 0.0375   if epoch == 2
        >>> # lr = 0.04375  if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = LinearLR(self.opt, start_factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1,
                 verbose=False):
        if start_factor > 1.0 or start_factor < 0:
            raise ValueError('Starting multiplicative factor expected to be between 0 and 1.')

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError('Ending multiplicative factor expected to be between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super(LinearLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] * self.start_factor for group in self.optimizer.param_groups]

        if (self.last_epoch > self.total_iters):
            return [group['lr'] for group in self.optimizer.param_groups]

        return [group['lr'] * (1. + (self.end_factor - self.start_factor) /
                (self.total_iters * self.start_factor + (self.last_epoch - 1) * (self.end_factor - self.start_factor)))
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * (self.start_factor +
                (self.end_factor - self.start_factor) * min(self.total_iters, self.last_epoch) / self.total_iters)
                for base_lr in self.base_lrs]

class SequentialLR(torch.optim.lr_scheduler._LRScheduler):
    """Receives the list of schedulers that is expected to be called sequentially during
    optimization process and milestone points that provides exact intervals to reflect
    which scheduler is supposed to be called at a given epoch.
    Args:
        schedulers (list): List of chained schedulers.
        milestones (list): List of integers that reflects milestone points.
    Example:
        >>> # Assuming optimizer uses lr = 1. for all groups
        >>> # lr = 0.1     if epoch == 0
        >>> # lr = 0.1     if epoch == 1
        >>> # lr = 0.9     if epoch == 2
        >>> # lr = 0.81    if epoch == 3
        >>> # lr = 0.729   if epoch == 4
        >>> scheduler1 = ConstantLR(self.opt, factor=0.1, total_iters=2)
        >>> scheduler2 = ExponentialLR(self.opt, gamma=0.9)
        >>> scheduler = SequentialLR(self.opt, schedulers=[scheduler1, scheduler2], milestones=[2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1, verbose=False):
        for scheduler_idx in range(1, len(schedulers)):
            if (schedulers[scheduler_idx].optimizer != schedulers[0].optimizer):
                raise ValueError(
                    "Sequential Schedulers expects all schedulers to belong to the same optimizer, but "
                    "got schedulers at index {} and {} to be different".format(0, scheduler_idx)
                )
        if (len(milestones) != len(schedulers) - 1):
            raise ValueError(
                "Sequential Schedulers expects number of schedulers provided to be one more "
                "than the number of milestone points, but got number of schedulers {} and the "
                "number of milestones to be equal to {}".format(len(schedulers), len(milestones))
            )
        self._schedulers = schedulers
        self._milestones = milestones
        self.last_epoch = last_epoch + 1
        self.optimizer = optimizer

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            self._schedulers[idx].step(0)
        else:
            self._schedulers[idx].step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The wrapped scheduler states will also be saved.
        """
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', '_schedulers')}
        state_dict['_schedulers'] = [None] * len(self._schedulers)

        for idx, s in enumerate(self._schedulers):
            state_dict['_schedulers'][idx] = s.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        _schedulers = state_dict.pop('_schedulers')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['_schedulers'] = _schedulers

        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)

class ConstantLR(torch.optim.lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group by a small constant factor until the
    number of epoch reaches a pre-defined milestone: total_iters. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside this scheduler.
    When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        factor (float): The number we multiply learning rate until the milestone. Default: 1./3.
        total_iters (int): The number of steps that the scheduler decays the learning rate.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025   if epoch == 0
        >>> # lr = 0.025   if epoch == 1
        >>> # lr = 0.025   if epoch == 2
        >>> # lr = 0.025   if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = ConstantLR(self.opt, factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, factor=1.0 / 3, total_iters=5, last_epoch=-1, verbose=False):
        if factor > 1.0 or factor < 0:
            raise ValueError('Constant multiplicative factor expected to be between 0 and 1.')

        self.factor = factor
        self.total_iters = total_iters
        super(ConstantLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] * self.factor for group in self.optimizer.param_groups]

        if (self.last_epoch > self.total_iters or
                (self.last_epoch != self.total_iters)):
            return [group['lr'] for group in self.optimizer.param_groups]

        if (self.last_epoch == self.total_iters):
            return [group['lr'] * (1.0 / self.factor) for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * (self.factor + (self.last_epoch >= self.total_iters) * (1 - self.factor))
                for base_lr in self.base_lrs]

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.inference_mode():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return ("global correct: {:.1f}\naverage row correct: {}\nIoU: {}\nmean IoU: {:.1f}").format(
            acc_global.item() * 100,
            [f"{i:.1f}" for i in (acc * 100).tolist()],
            [f"{i:.1f}" for i in (iu * 100).tolist()],
            iu.mean().item() * 100,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str}")


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs


# def collate_fn(batch):
#     images, targets = list(zip(*batch))
#     batched_imgs = cat_list(images, fill_value=0)
#     batched_targets = cat_list(targets, fill_value=255)
#     return batched_imgs, batched_targets

def collate_fn(batch):
    images, targets, fname = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)

    return batched_imgs, batched_targets, fname

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    args.rank = 0#int(os.environ["RANK"])
    args.world_size = 2#int(os.environ["WORLD_SIZE"])
    args.gpu = 0#int(os.environ["LOCAL_RANK"])
    # elif "SLURM_PROCID" in os.environ:
    #     args.rank = int(os.environ["SLURM_PROCID"])
    #     args.gpu = args.rank % torch.cuda.device_count()
    # elif hasattr(args, "rank"):
    #     pass
    # else:
    #     print("Not using distributed mode")
    #     args.distributed = False
    #     return

    # if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    #     args.rank = int(os.environ["RANK"])
    #     args.world_size = int(os.environ["WORLD_SIZE"])
    #     args.gpu = int(os.environ["LOCAL_RANK"])
    # elif "SLURM_PROCID" in os.environ:
    #     args.rank = int(os.environ["SLURM_PROCID"])
    #     args.gpu = args.rank % torch.cuda.device_count()
    # elif hasattr(args, "rank"):
    #     pass
    # else:
    #     print("Not using distributed mode")
    #     args.distributed = False
    #     return

    args.distributed = True

    torch.cuda.set_device(args.device_ids[0])
    args.dist_backend = args.dist_backend
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    setup_for_distributed(args.rank == 0)
