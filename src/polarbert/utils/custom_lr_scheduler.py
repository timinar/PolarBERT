from torch.optim.lr_scheduler import LRScheduler, _warn_get_lr_called_within_step
from torch.optim import Optimizer

# Inspired from LinearLR and OneCycleLR from PyTorch
class TrapezoidalLR(LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        decay_steps: int,
        total_steps=None,
        epochs=None,
        steps_per_epoch=None,
        last_epoch: int = -1,
    ):
        # Validate optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer

        # Validate warmup_steps and decay_steps
        if not isinstance(warmup_steps, int) or warmup_steps < 0:
            raise ValueError(f"Expected non-negative integer warmup_steps, but got {warmup_steps}")
        if not isinstance(decay_steps, int) or decay_steps < 0:
            raise ValueError(f"Expected non-negative integer decay_steps, but got {decay_steps}")
        
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

        # Validate total_steps
        if total_steps is not None:
            if total_steps <= 0 or not isinstance(total_steps, int):
                raise ValueError(
                    f"Expected positive integer total_steps, but got {total_steps}"
                )
            self.total_steps = total_steps
        elif epochs is not None and steps_per_epoch is not None:
            if not isinstance(epochs, int) or epochs <= 0:
                raise ValueError(f"Expected positive integer epochs, but got {epochs}")
            if not isinstance(steps_per_epoch, int) or steps_per_epoch <= 0:
                raise ValueError(
                    f"Expected positive integer steps_per_epoch, but got {steps_per_epoch}"
                )
            self.total_steps = epochs * steps_per_epoch
        else:
            raise ValueError(
                "You must define either total_steps OR (epochs AND steps_per_epoch)"
            )

        # Validate schedule consistency
        if self.warmup_steps + self.decay_steps > self.total_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) + decay_steps ({self.decay_steps}) = "
                f"{self.warmup_steps + self.decay_steps} exceeds total_steps ({self.total_steps})"
            )

        self.decay_start = self.total_steps - self.decay_steps

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute the learning rate."""
        _warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            # For the first step, scale from base_lr to the target lr at step 0
            scaling_factor = self._get_scaling_factor(0)
            result = [group["lr"] * scaling_factor for group in self.optimizer.param_groups]
            
            return result
        
        # For subsequent steps, compute the multiplicative factor: scaling(n) / scaling(n-1)
        current_scaling = self._get_scaling_factor(self.last_epoch)
        previous_scaling = self._get_scaling_factor(self.last_epoch - 1)
        
        if previous_scaling == 0.0:
            # Handle division by zero
            if current_scaling == 0.0:
                # Both are zero, no change needed
                return [group["lr"] for group in self.optimizer.param_groups]
            else:
                # Previous was zero, current is non-zero - this indicates invalid schedule configuration
                raise ValueError(f"Invalid learning rate schedule: scaling factor jumps from 0.0 to {current_scaling:.3g} between steps {self.last_epoch-1} and {self.last_epoch}.")
        else:
            # Normal case: multiply by the ratio of scaling factors
            multiplicative_factor = current_scaling / previous_scaling
            result = [group["lr"] * multiplicative_factor for group in self.optimizer.param_groups]
            
            return result
    
    def _get_scaling_factor(self, step: int) -> float:
        if step < 0:
            scaling_factor = 0.0
        elif self.warmup_steps > 0 and step < self.warmup_steps:
            scaling_factor = (step + 1) / self.warmup_steps
        elif step < self.decay_start:
            scaling_factor = 1.0
        elif self.decay_steps > 0 and step <= self.total_steps:
            scaling_factor = (self.total_steps - step) / self.decay_steps
        else:
            scaling_factor = 0.0
        return scaling_factor

    def _get_closed_form_lr(self):
        return [
            base_lr
            * self._get_scaling_factor(self.last_epoch)
            for base_lr in self.base_lrs
        ]