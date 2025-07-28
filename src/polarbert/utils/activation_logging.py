import csv
from pathlib import Path
from typing import Optional, Any

import torch
import pytorch_lightning as pl


class ActivationLoggingCallback(pl.Callback):
    """
    PyTorch Lightning callback for logging mean absolute activation values.
    
    This callback registers forward hooks on all named modules to compute and log
    mean absolute activation values. It supports logging to both W&B and CSV files.
    """
    
    def __init__(
        self, 
        log_frequency: int = 100,
        csv_path: Optional[str] = None
    ):
        """
        Initialize the activation logging callback.
        
        Args:
            log_frequency: How often to log activations (in training steps)
            csv_path: Optional path to CSV file for logging activation stats
        """
        super().__init__()
        self.log_frequency = log_frequency
        self.csv_path = csv_path
        self.hooks = []
        self.activation_stats = {}
        self.step_count = 0
        self.csv_file = None
        self.csv_writer = None
        
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Register forward hooks on all named modules when training starts."""
        self._register_hooks(pl_module)
        self._setup_csv_logging()
        
    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Ensure hooks are registered for validation as well."""
        if not self.hooks:  # Only register if not already registered
            self._register_hooks(pl_module)
            
    def on_train_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: Any, 
        batch: Any, 
        batch_idx: int
    ) -> None:
        """Log activation statistics at specified frequency during training."""
        self.step_count += 1
        if self.step_count % self.log_frequency == 0:
            self._log_activations(trainer, "train")
            
    def on_validation_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: Any, 
        batch: Any, 
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        """Log activation statistics during validation."""
        # Log less frequently during validation to avoid clutter
        if batch_idx % self.log_frequency == 0:
            self._log_activations(trainer, "val")
            
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Clean up hooks and close CSV file when training ends."""
        self._cleanup_hooks()
        self._cleanup_csv_logging()

    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: Exception) -> None:
        """Clean up hooks and close CSV file when an exception occurs."""
        self._cleanup_hooks()
        self._cleanup_csv_logging()
        
    def _register_hooks(self, model: torch.nn.Module) -> None:
        """Register forward hooks on all named modules."""
        for name, module in model.named_modules():
            if name:  # Skip the root module (empty name)
                hook = module.register_forward_hook(
                    self._create_hook_fn(name)
                )
                self.hooks.append(hook)
                
    # Supported numeric tensor types for activation logging
    _NUMERIC_TYPES = (torch.float16, torch.float32, torch.float64, torch.bfloat16)
    
    def _create_hook_fn(self, module_name: str):
        """Create a forward hook function for a specific module."""
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                # Only compute mean for numeric tensors (skip boolean/integer masks)
                if output.dtype in self._NUMERIC_TYPES:
                    mean_abs_activation = torch.mean(torch.abs(output.detach())).item()
                    self.activation_stats[module_name] = mean_abs_activation
            elif isinstance(output, (tuple, list)):
                # Handle modules that return multiple tensors (like some attention modules)
                for i, out_tensor in enumerate(output):
                    if isinstance(out_tensor, torch.Tensor):
                        # Only compute mean for numeric tensors (skip boolean/integer masks)
                        if out_tensor.dtype in self._NUMERIC_TYPES:
                            mean_abs_activation = torch.mean(torch.abs(out_tensor.detach())).item()
                            self.activation_stats[f"{module_name}.output_{i}"] = mean_abs_activation
        return hook_fn
        
    def _log_activations(self, trainer: pl.Trainer, phase: str) -> None:
        """Log activation statistics to W&B and optionally to CSV."""
        if not self.activation_stats:
            return
            
        # Log to W&B if logger is available
        if hasattr(trainer.logger, 'experiment') and trainer.logger.experiment is not None:
            wandb_logs = {}
            for module_name, mean_abs_act in self.activation_stats.items():
                wandb_logs[f"mean_abs_activation/{module_name}"] = mean_abs_act
            trainer.logger.experiment.log(wandb_logs, step=trainer.global_step)
            
        # Log to CSV if configured
        if self.csv_writer is not None:
            for module_name, mean_abs_act in self.activation_stats.items():
                self.csv_writer.writerow({
                    'step': trainer.global_step,
                    'phase': phase,
                    'module_name': module_name,
                    'mean_abs_activation': mean_abs_act
                })
            self.csv_file.flush()  # Ensure data is written to disk
            
        # Clear stats for next batch
        self.activation_stats.clear()
        
    def _setup_csv_logging(self) -> None:
        """Set up CSV file for logging if path is provided."""
        if self.csv_path is None:
            return
            
        # Create directory if it doesn't exist
        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Open CSV file and set up writer
        self.csv_file = open(self.csv_path, 'w', newline='')
        fieldnames = ['step', 'phase', 'module_name', 'mean_abs_activation']
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        
    def _cleanup_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def _cleanup_csv_logging(self) -> None:
        """Close CSV file if it was opened."""
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None 