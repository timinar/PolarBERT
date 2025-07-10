# config.py
import yaml
import os
import warnings
import math # Needed for ceil
from typing import List, Optional, Dict, Any, Tuple

# --- Configuration Classes ---

class DataConfig:
    def __init__(self, data: Dict[str, Any]):
        # Define expected keys for this config section
        expected_keys = {
            'max_per_device_batch_size', 'train_dir', 'val_dir',
            'train_events', 'val_events', 'pin_memory',
            'num_workers', 'persistent_workers'
        }
        self._validate_keys("DataConfig", data.keys(), expected_keys)

        try:
            self.max_per_device_batch_size: int = int(data.get('max_per_device_batch_size', 4096))
            self.train_dir: str = str(data.get('train_dir', '/path/to/train/data'))
            self.val_dir: str = str(data.get('val_dir', '/path/to/val/data'))
            self.prometheus_dir: Optional[str] = str(data.get('prometheus_dir', '/path/to/prometheus/data'))
            train_events_val = data.get('train_events', 100_000_000)
            self.train_events: Optional[int] = int(train_events_val) if train_events_val is not None else None
            val_events_val = data.get('val_events', 200_000)
            self.val_events: Optional[int] = int(val_events_val) if val_events_val is not None else None
            self.pin_memory: bool = bool(data.get('pin_memory', False))
            self.num_workers: int = int(data.get('num_workers', 1))
            self.persistent_workers: bool = bool(data.get('persistent_workers', True))
        except (ValueError, TypeError) as e:
            print(f"Error parsing DataConfig: {e}. Input data: {data}")
            raise

    def _validate_keys(self, class_name: str, provided_keys: set, expected_keys: set):
        """Warns about unexpected keys in the config dictionary."""
        unexpected_keys = set(provided_keys) - expected_keys
        if unexpected_keys:
            warnings.warn(
                f"In {class_name}: Unexpected keys found in config dictionary and will be ignored: {unexpected_keys}",
                UserWarning
            )

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

class EmbeddingConfig:
    def __init__(self, data: Dict[str, Any]):
        expected_keys = {
            'time_embedding_dim', 'dom_embedding_dim', 'charge_embedding_dim', 'aux_embedding_dim',
            'embedding_dim', 'time_vocab_size', 'dom_vocab_size', 'charge_vocab_size',
            'charge_bin_min', 'charge_bin_max', #'aux_vocab_size', # Example if needed
            'masking_doms', 'masking_times', 'masking_charges', 'masking_prob',
            'embedding_projection'
        }
        self._validate_keys("EmbeddingConfig", data.keys(), expected_keys)

        try:
            self.time_embedding_dim: int = int(data.get('time_embedding_dim', 128))
            self.dom_embedding_dim: int = int(data.get('dom_embedding_dim', 108))
            self.charge_embedding_dim: int = int(data.get('charge_embedding_dim', 16))
            self.aux_embedding_dim: int = int(data.get('aux_embedding_dim', 4))
            self.embedding_dim: int = int(data.get('embedding_dim', 256)) # Overall target dim

            self.time_vocab_size: int = int(data.get('time_vocab_size', 52000))
            self.dom_vocab_size: int = int(data.get('dom_vocab_size', 5162))
            self.charge_vocab_size: int = int(data.get('charge_vocab_size', 128))
            self.charge_bin_min: float = float(data.get('charge_bin_min', -0.6))
            self.charge_bin_max: float = float(data.get('charge_bin_max', 0.9))
            # self.aux_vocab_size: int = int(data.get('aux_vocab_size', 4)) # If needed

            self.masking_doms: bool = bool(data.get('masking_doms', True))
            self.masking_times: bool = bool(data.get('masking_times', False))
            self.masking_charges: bool = bool(data.get('masking_charges', False))
            self.masking_prob: float = float(data.get('masking_prob', 0.25)) # Kept here
            self.embedding_projection: bool = bool(data.get('embedding_projection', False))

            self._sum_sub_dims = (
                self.time_embedding_dim + self.dom_embedding_dim +
                self.charge_embedding_dim + self.aux_embedding_dim
            )
        except (ValueError, TypeError) as e:
            print(f"Error parsing EmbeddingConfig: {e}. Input data: {data}")
            raise

    def _validate_keys(self, class_name: str, provided_keys: set, expected_keys: set):
        """Warns about unexpected keys in the config dictionary."""
        unexpected_keys = set(provided_keys) - expected_keys
        if unexpected_keys:
            warnings.warn(
                f"In {class_name}: Unexpected keys found in config dictionary and will be ignored: {unexpected_keys}",
                UserWarning
            )

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d.pop('_sum_sub_dims', None) # Don't save derived value
        return d

class ModelConfig:
    def __init__(self, data: Dict[str, Any]):
        # Keys expected at the top level of the 'model' section in YAML
        # Includes keys for nested configs ('embedding', 'directional_head', etc.)
        expected_keys = {
            'embedding_dim', 'num_heads', 'hidden_size', 'num_layers',
            'ffd_type', 'lambda_charge', 'dropout', 'norm_eps',
            'model_name', 'use_rope', 'use_positional_embedding',
            'embedding', # Key for nested EmbeddingConfig dict
            'directional_head', # Example key for nested head config dict
            'energy_head' # Example key for nested head config dict
        }
        # Note: We validate top-level keys here. Nested dicts ('embedding', 'directional_head')
        # will be validated by their respective constructors if they exist.
        self._validate_keys("ModelConfig", data.keys(), expected_keys)

        try:
            self.embedding_dim: int = int(data.get('embedding_dim', 256))
            self.num_heads: int = int(data.get('num_heads', 8))
            self.hidden_size: int = int(data.get('hidden_size', 1024))
            self.num_layers: int = int(data.get('num_layers', 8))
            self.ffd_type: str = str(data.get('ffd_type', 'SwiGLU'))
            # lambda_charge is for pre-training, might be removed or ignored in fine-tuning specific config
            self.lambda_charge: float = float(data.get('lambda_charge', 1.0))
            self.dropout: float = float(data.get('dropout', 0.1)) # Adjusted default
            self.norm_eps: float = float(data.get('norm_eps', 1e-5))
            self.model_name: str = str(data.get('model_name', 'polarbert_model'))
            self.use_rope: bool = bool(data.get('use_rope', False))
            self.use_positional_embedding: bool = bool(data.get('use_positional_embedding', False))

            # Handle nested embedding config
            embedding_data = data.get('embedding', {})
            self.embedding = EmbeddingConfig(embedding_data)
            # Ensure model's embedding_dim matches embedding config's target dim if projection is used
            # Or ensure it matches the sum if projection is False (validation happens later)
            self.embedding_dim = self.embedding.embedding_dim # Use the dim from embedding config

            # Handle potential nested head configs (example for directional)
            # The fine-tuning script might access these directly via config.model.<head_name>.parameter
            # Store the raw dictionary; parsing can happen in the model using it
            self.directional_head: Optional[Dict[str, Any]] = data.get('directional_head', None)
            self.energy_head: Optional[Dict[str, Any]] = data.get('energy_head', None)


        except (ValueError, TypeError) as e:
            print(f"Error parsing ModelConfig: {e}. Input data: {data}")
            raise

    def _validate_keys(self, class_name: str, provided_keys: set, expected_keys: set):
        """Warns about unexpected keys in the config dictionary."""
        # Convert provided_keys to set if it's dict_keys
        unexpected_keys = set(provided_keys) - expected_keys
        if unexpected_keys:
            warnings.warn(
                f"In {class_name}: Unexpected keys found in config dictionary and will be ignored: {unexpected_keys}",
                UserWarning
            )

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d['embedding'] = self.embedding.to_dict()
        # Keep nested head dicts as they are, or implement to_dict for them if they become classes
        # d['directional_head'] = self.directional_head.to_dict() # If it were a class
        return d


class CheckpointConfig:
    def __init__(self, data: Dict[str, Any]):
        expected_keys = {'dirpath', 'save_top_k', 'monitor', 'mode', 'save_last', 'save_final'}
        self._validate_keys("CheckpointConfig", data.keys(), expected_keys)
        try:
            self.dirpath: str = str(data.get('dirpath', 'checkpoints'))
            self.save_top_k: int = int(data.get('save_top_k', -1))
            self.monitor: str = str(data.get('monitor', 'val/loss')) # Default fine-tuning monitor
            self.mode: str = str(data.get('mode', 'min'))
            self.save_last: bool = bool(data.get('save_last', True))
            self.save_final: bool = bool(data.get('save_final', False)) # Default false for fine-tuning
        except (ValueError, TypeError) as e:
            print(f"Error parsing CheckpointConfig: {e}. Input data: {data}")
            raise

    def _validate_keys(self, class_name: str, provided_keys: set, expected_keys: set):
        """Warns about unexpected keys in the config dictionary."""
        unexpected_keys = set(provided_keys) - expected_keys
        if unexpected_keys:
            warnings.warn(
                f"In {class_name}: Unexpected keys found in config dictionary and will be ignored: {unexpected_keys}",
                UserWarning
            )

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

class LoggingConfig:
    def __init__(self, data: Dict[str, Any]):
        expected_keys = {'project'} # Add 'entity', 'log_every_n_steps' etc. if used
        self._validate_keys("LoggingConfig", data.keys(), expected_keys)
        try:
            self.project: str = str(data.get('project', 'PolarBERT-Default-Project'))
            # Add other logging args here
        except (ValueError, TypeError) as e:
            print(f"Error parsing LoggingConfig: {e}. Input data: {data}")
            raise

    def _validate_keys(self, class_name: str, provided_keys: set, expected_keys: set):
        """Warns about unexpected keys in the config dictionary."""
        unexpected_keys = set(provided_keys) - expected_keys
        if unexpected_keys:
            warnings.warn(
                f"In {class_name}: Unexpected keys found in config dictionary and will be ignored: {unexpected_keys}",
                UserWarning
            )

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class TrainingConfig:
    def __init__(self, data: Dict[str, Any]):
        try:
            # --- Training loop ---
            self.max_epochs: int = int(data.get('max_epochs', 20))
            self.logical_batch_size: int = int(data.get('logical_batch_size', 4096))
            self.val_check_interval: float = float(data.get('val_check_interval', 1.0)) # Default to 1 epoch
            self.gpus: int = int(data.get('gpus', 1))
            self.precision: str = str(data.get('precision', '16-mixed'))
            self.gradient_clip_val: float = float(data.get('gradient_clip_val', 1.0))

            # --- Fine-tuning / Multi-Task Specific ---
            self.task: str = str(data.get('task', 'direction')) # Primary task
            # Default 'mean'. Options: 'mean', 'cls'
            self.directional_pooling_mode: str = str(data.get('directional_pooling_mode', 'mean')).lower()
             # Default 1.0. Weight for auxiliary DOM loss during multi-task finetuning
            self.lambda_dom: float = float(data.get('lambda_dom', 1.0))
            # Add freeze_backbone and pretrained_checkpoint_path if they should be part of the static config
            # Or handle them purely via command-line args and pass to model init separately
            self.freeze_backbone: bool = bool(data.get('freeze_backbone', False))
            self.pretrained_checkpoint_path: Optional[str] = data.get('pretrained_checkpoint_path', None) # Record path from config if provided

            # --- Optimizer ---
            self.optimizer: str = str(data.get('optimizer', 'AdamW'))
            self.max_lr: float = float(data.get('max_lr', 3e-4))
            self.adam_beta1: float = float(data.get('adam_beta1', 0.9))
            self.adam_beta2: float = float(data.get('adam_beta2', 0.95))
            self.adam_eps: float = float(data.get('adam_eps', 1e-8))
            self.weight_decay: float = float(data.get('weight_decay', 0.1))
            self.amsgrad: bool = bool(data.get('amsgrad', False))

            # --- Scheduler ---
            self.lr_scheduler: str = str(data.get('lr_scheduler', 'onecycle'))
            self.warm_up_steps: Optional[int] = data.get('warm_up_steps', 1000) # Allow None explicitly
            if self.warm_up_steps is not None: self.warm_up_steps = int(self.warm_up_steps)
            self.pct_start: float = float(data.get('pct_start', 0.1)) # Adjusted default, may be overridden
            self.div_factor: float = float(data.get('div_factor', 25.0))
            self.final_div_factor: float = float(data.get('final_div_factor', 1e4))

            # --- Nested configs ---
            checkpoint_data = data.get('checkpoint', {})
            self.checkpoint = CheckpointConfig(checkpoint_data)
            logging_data = data.get('logging', {})
            self.logging = LoggingConfig(logging_data)

            # --- Runtime calculated attributes ---
            self.steps_per_epoch: Optional[int] = None
            self.total_steps: Optional[int] = None
            self.per_device_batch_size: Optional[int] = None
            self.gradient_accumulation_steps: Optional[int] = None
            self.effective_batch_size: Optional[int] = None

            # --- Validate pooling mode ---
            if self.directional_pooling_mode not in ['mean', 'cls']:
                 raise ValueError(f"Invalid directional_pooling_mode: '{self.directional_pooling_mode}'. Must be 'mean' or 'cls'.")


        except (ValueError, TypeError) as e:
            print(f"Error parsing TrainingConfig: {e}. Input data: {data}")
            raise

    def calculate_runtime_params(self, num_batches_per_epoch: int, max_per_device_batch_size: int):
        """Calculates derived parameters based on dataloader length and batch sizes."""
        if num_batches_per_epoch <= 0:
             # Allow calculation even if estimate is 0, but warn heavily if total_steps needed.
             warnings.warn("num_batches_per_epoch is zero or negative. Step calculations might be incorrect.")
             num_batches_per_epoch = 1 # Avoid division by zero later, but result is likely wrong

        print("Calculating runtime training parameters...")
        # --- Batch size calculation ---
        logical_batch = self.logical_batch_size
        self.per_device_batch_size = min(max_per_device_batch_size, logical_batch)
        if self.per_device_batch_size == 0: raise ValueError("Calculated per_device_batch_size is zero.") # Prevent division by zero
        self.gradient_accumulation_steps = math.ceil(logical_batch / self.per_device_batch_size)
        self.effective_batch_size = self.gradient_accumulation_steps * self.per_device_batch_size
        print(f"  Logical Batch Size: {self.logical_batch_size}")
        print(f"  Max Per Device Batch Size: {max_per_device_batch_size}")
        print(f"  Calculated Per Device Batch Size: {self.per_device_batch_size}")
        print(f"  Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        print(f"  Effective Batch Size: {self.effective_batch_size}")

        # --- Step calculation ---
        # num_batches_per_epoch is batches using per_device_batch_size
        optimizer_steps_per_epoch = math.ceil(num_batches_per_epoch / self.gradient_accumulation_steps)
        # Total steps usually refers to optimizer steps for schedulers like OneCycleLR
        self.total_steps = optimizer_steps_per_epoch * self.max_epochs
        self.steps_per_epoch = optimizer_steps_per_epoch # Store optimizer steps per epoch
        print(f"  Optimizer Steps per Epoch: {self.steps_per_epoch}")
        print(f"  Total Optimizer Steps: {self.total_steps}")

        # --- Adjust pct_start based on warm_up_steps ---
        # Use total OPTIMIZER steps for scheduler calculations
        if self.warm_up_steps is not None and self.warm_up_steps > 0:
            if self.total_steps > 0:
                 # pct_start should be fraction of total optimizer steps
                 calculated_pct_start = min(1.0, self.warm_up_steps / self.total_steps)
                 # Only override if warm_up_steps was explicitly provided
                 # Check if the calculated value differs significantly from config default/value
                 # if abs(calculated_pct_start - self.pct_start) > 1e-5:
                 print(f"  Overriding pct_start based on warm_up_steps: "
                       f"{self.pct_start:.4f} -> {calculated_pct_start:.4f} "
                       f"(Warmup: {self.warm_up_steps}, Total Steps: {self.total_steps})")
                 self.pct_start = calculated_pct_start
            else:
                 print("  Warning: Cannot calculate pct_start from warm_up_steps as total_steps is zero.")


    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d['checkpoint'] = self.checkpoint.to_dict()
        d['logging'] = self.logging.to_dict()
        # Exclude runtime calculated fields if they shouldn't be saved in YAML
        # Or keep them if they are useful for reproducibility
        # runtime_keys = ['steps_per_epoch', 'total_steps', 'per_device_batch_size',
        #                 'gradient_accumulation_steps', 'effective_batch_size']
        # for key in runtime_keys:
        #     d.pop(key, None)
        return d

# --- Main Config Class ---
class PolarBertConfig: # Renamed from ExperimentConfig
    """Main configuration class orchestrating all sub-configurations."""
    def __init__(self, data_cfg: Dict, model_cfg: Dict, training_cfg: Dict):
        self.data = DataConfig(data_cfg)
        self.model = ModelConfig(model_cfg)
        # Pass the model config embedding dim check to TrainingConfig if needed later
        self.training = TrainingConfig(training_cfg)
        self._validate() # Call validation method on initialization

    @classmethod
    def from_yaml(cls, path: str) -> 'PolarBertConfig':
        """Loads configuration from a YAML file."""
        # ... (previous code for loading YAML) ...
        print(f"Loading configuration from: {path}")
        try:
            with open(path, 'r') as f:
                 cfg_dict = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {path}")
            raise
        except yaml.YAMLError as e:
            print(f"Error: Could not parse YAML file at {path}: {e}")
            raise

        required_keys = ['data', 'model', 'training']
        if not all(key in cfg_dict for key in required_keys):
             raise ValueError(f"YAML file must contain top-level keys: {required_keys}")
        if 'logging' not in cfg_dict.get('training', {}):
             warnings.warn("YAML training section does not contain 'logging' subsection.")
        if 'checkpoint' not in cfg_dict.get('training', {}):
             warnings.warn("YAML training section does not contain 'checkpoint' subsection.")

        return cls(
            data_cfg=cfg_dict.get('data', {}),
            model_cfg=cfg_dict.get('model', {}),
            training_cfg=cfg_dict.get('training', {})
        )


    def calculate_runtime_params(self, num_batches_per_epoch: int):
         """Calculates runtime parameters (steps, batches) after dataloader is known."""
         # Pass the necessary value from DataConfig to TrainingConfig calculation method
         self.training.calculate_runtime_params(num_batches_per_epoch, self.data.max_per_device_batch_size)


    def to_dict(self) -> Dict[str, Any]:
         # This will now include lambda_dom and directional_pooling_mode from TrainingConfig
         return {
             'data': self.data.to_dict(),
             'model': self.model.to_dict(),
             'training': self.training.to_dict()
         }

    def save_yaml(self, path: str):
        """Saves the current configuration to a YAML file."""
        # ... (previous code for saving YAML) ...
        print(f"Saving configuration to: {path}")
        target_dir = os.path.dirname(path)
        if target_dir: # Only create if path includes a directory
             os.makedirs(target_dir, exist_ok=True)
        # Get dict WITH calculated runtime params included
        cfg_dict = self.to_dict()
        try:
            with open(path, 'w') as f:
                 yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
        except IOError as e:
            print(f"Error: Could not write config file to {path}: {e}")
            raise


    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config_filename="config.yaml") -> 'PolarBertConfig':
        """Loads configuration from a YAML file in the same directory as a checkpoint."""
        # ... (previous code) ...
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        config_path = os.path.join(os.path.dirname(checkpoint_path), config_filename)
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                 f"Config file '{config_filename}' not found in checkpoint directory: "
                 f"{os.path.dirname(checkpoint_path)}"
            )
        print(f"Loading config associated with checkpoint {os.path.basename(checkpoint_path)}")
        return cls.from_yaml(config_path)


    def _validate(self):
         """Performs configuration validation checks."""
         print("Validating configuration...")
         # --- Model Validation ---
         if self.model.embedding_dim % self.model.num_heads != 0:
             raise ValueError(f"model.embedding_dim ({self.model.embedding_dim}) must be divisible "
                              f"by model.num_heads ({self.model.num_heads})")

         # Validate RoPE and positional embedding configuration
         if self.model.use_rope and self.model.use_positional_embedding:
             raise ValueError("model.use_positional_embedding must be False when model.use_rope is True")

         # --- Embedding Validation ---
         emb_cfg = self.model.embedding
         if not emb_cfg.embedding_projection:
             if emb_cfg._sum_sub_dims != self.model.embedding_dim:
                 raise ValueError(
                     f"embedding.embedding_projection is False, but sum of sub-embedding dims "
                     f"({emb_cfg._sum_sub_dims}) does not match model.embedding_dim "
                     f"({self.model.embedding_dim}). Adjust dimensions or set projection to True."
                 )

         # --- Training Validation ---
         train_cfg = self.training
         # Check pooling mode validity (already done in TrainingConfig.__init__)
         # if train_cfg.directional_pooling_mode not in ['mean', 'cls']:
         #     raise ValueError(...)

         # Check final_div_factor only if scheduler is onecycle
         if isinstance(train_cfg.lr_scheduler, str) and train_cfg.lr_scheduler.lower() == 'onecycle':
             if not isinstance(train_cfg.final_div_factor, (int, float)):
                 raise TypeError(f"training.final_div_factor must be a number, got {type(train_cfg.final_div_factor)}")
             if train_cfg.final_div_factor < 1.0:
                 warnings.warn(f"training.final_div_factor ({train_cfg.final_div_factor}) < 1.0. "
                               f"Usually >= 1.0 (e.g., 1e4) for OneCycleLR.", UserWarning)

         # Warn if warm_up_steps and pct_start might conflict (handled by recalculation logic now)
         # if train_cfg.warm_up_steps is not None and train_cfg.warm_up_steps > 0:
         #     # Check if pct_start differs significantly from default only if warm_up_steps is used
         #     if not math.isclose(train_cfg.pct_start, 0.1): # Default check
         #          warnings.warn(f"training.warm_up_steps ({train_cfg.warm_up_steps}) is set. "
         #                      f"Explicit training.pct_start ({train_cfg.pct_start}) might be ignored.", UserWarning)

         print("Configuration validation passed (with potential warnings).")

# --- Example Usage (Illustrative) ---
# config_file = "/path/to/your/finetuning_config.yaml"
# try:
#     config = PolarBertConfig.from_yaml(config_file)
#     # Mock calculation needs estimate of batches per epoch
#     num_batches_per_epoch = 10000 # Placeholder! Calculate properly in main script.
#     config.calculate_runtime_params(num_batches_per_epoch)
#     print(config.training.directional_pooling_mode)
#     print(config.training.lambda_dom)
# except Exception as e:
#      print(f"Error: {e}")