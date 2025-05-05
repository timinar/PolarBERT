# src/polarbert/config.py
import yaml
import os
import warnings
import math # Needed for ceil
from typing import List, Optional, Dict, Any, Tuple, Union # Added Union

# --- Configuration Classes ---

class DataConfig:
    def __init__(self, data: Dict[str, Any]):
        # Define expected keys for this config section
        expected_keys = {
            # General paths and batching
            'max_per_device_batch_size',
            'pin_memory', 'num_workers', 'persistent_workers',
            # Kaggle paths and event counts (using general names)
            'train_dir', 'val_dir',
            'train_events', 'val_events',
            # Prometheus path and event counts (specific names)
            'prometheus_dir',
            'prometheus_train_events', 'prometheus_val_events',
        }
        self._validate_keys("DataConfig", data.keys(), expected_keys)

        try:
            # Batching
            self.max_per_device_batch_size: int = int(data.get('max_per_device_batch_size', 1024))

            # Paths
            self.train_dir: str = str(data.get('train_dir', '/path/to/kaggle/train/data')) # Primary (Kaggle) Train
            self.val_dir: str = str(data.get('val_dir', '/path/to/kaggle/val/data'))       # Primary (Kaggle) Validation
            self.prometheus_dir: Optional[str] = data.get('prometheus_dir', None)         # Prometheus Base

            # Event Counts
            # Primary (Kaggle) counts - use None to signify using the full dataset in the directory
            train_events_val = data.get('train_events', None)
            self.train_events: Optional[int] = int(train_events_val) if train_events_val is not None else None
            val_events_val = data.get('val_events', None)
            self.val_events: Optional[int] = int(val_events_val) if val_events_val is not None else None

            # Prometheus counts (Required for manual splitting)
            p_train_events_val = data.get('prometheus_train_events', None)
            self.prometheus_train_events: Optional[int] = int(p_train_events_val) if p_train_events_val is not None else None
            p_val_events_val = data.get('prometheus_val_events', None)
            self.prometheus_val_events: Optional[int] = int(p_val_events_val) if p_val_events_val is not None else None

            # DataLoader settings
            self.pin_memory: bool = bool(data.get('pin_memory', True))
            self.num_workers: int = int(data.get('num_workers', 1))
            self.persistent_workers: bool = bool(data.get('persistent_workers', self.num_workers > 0))

        except (ValueError, TypeError) as e:
            print(f"Error parsing DataConfig: {e}. Input data: {data}")
            raise

    def _validate_keys(self, class_name: str, provided_keys: set, expected_keys: set):
        """Warns about unexpected keys in the config dictionary."""
        provided_keys_set = set(provided_keys)
        unexpected_keys = provided_keys_set - expected_keys
        if unexpected_keys:
            warnings.warn(
                f"In {class_name}: Unexpected keys found in config dictionary and will be ignored: {unexpected_keys}",
                UserWarning
            )

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

# --- EmbeddingConfig, ModelConfig (No changes needed) ---
class EmbeddingConfig:
    # --- No changes needed here for mixed training ---
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
            self.dom_vocab_size: int = int(data.get('dom_vocab_size', 5162)) # 5160 DOMS + PAD + MASK
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
        provided_keys_set = set(provided_keys)
        unexpected_keys = provided_keys_set - expected_keys
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
    # --- No changes needed here for mixed training ---
    def __init__(self, data: Dict[str, Any]):
        expected_keys = {
            'embedding_dim', 'num_heads', 'hidden_size', 'num_layers',
            'ffd_type', 'lambda_charge', 'dropout', 'norm_eps',
            'model_name', 'use_rope', 'use_positional_embedding',
            'embedding', # Key for nested EmbeddingConfig dict
            'directional_head', # Key for nested head config dict
            'energy_head' # Key for nested head config dict
        }
        self._validate_keys("ModelConfig", data.keys(), expected_keys)

        try:
            self.embedding_dim: int = int(data.get('embedding_dim', 256))
            self.num_heads: int = int(data.get('num_heads', 8))
            self.hidden_size: int = int(data.get('hidden_size', 1024)) # Used in FFN
            self.num_layers: int = int(data.get('num_layers', 8))
            self.ffd_type: str = str(data.get('ffd_type', 'SwiGLU')) # 'SwiGLU' or 'MLP'
            self.lambda_charge: float = float(data.get('lambda_charge', 1.0)) # Weight for charge loss (pre-training)
            self.dropout: float = float(data.get('dropout', 0.0)) # Dropout rate (set to 0.0 usually)
            self.norm_eps: float = float(data.get('norm_eps', 1e-5)) # Epsilon for RMSNorm/LayerNorm
            self.model_name: str = str(data.get('model_name', 'polarbert_model')) # Base name for saving
            self.use_rope: bool = bool(data.get('use_rope', False)) # Rotary Positional Embeddings
            self.use_positional_embedding: bool = bool(data.get('use_positional_embedding', False)) # Standard learned pos embeds

            # Handle nested embedding config
            embedding_data = data.get('embedding', {})
            self.embedding = EmbeddingConfig(embedding_data)
            # Ensure model's embedding_dim matches embedding config's target dim if projection is used
            # Or ensure it matches the sum if projection is False (validation happens later)

            # Handle potential nested head configs (store raw dict, parse in module)
            self.directional_head: Optional[Dict[str, Any]] = data.get('directional_head', {'hidden_size': 1024}) # Add default structure
            self.energy_head: Optional[Dict[str, Any]] = data.get('energy_head', None)

        except (ValueError, TypeError) as e:
            print(f"Error parsing ModelConfig: {e}. Input data: {data}")
            raise

    def _validate_keys(self, class_name: str, provided_keys: set, expected_keys: set):
        """Warns about unexpected keys in the config dictionary."""
        provided_keys_set = set(provided_keys)
        unexpected_keys = provided_keys_set - expected_keys
        if unexpected_keys:
            warnings.warn(
                f"In {class_name}: Unexpected keys found in config dictionary and will be ignored: {unexpected_keys}",
                UserWarning
            )

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d['embedding'] = self.embedding.to_dict()
        # Keep nested head dicts as they are
        return d

class CheckpointConfig:
    # --- No changes needed here for mixed training, but adjust 'monitor' default ---
    def __init__(self, data: Dict[str, Any]):
        expected_keys = {'dirpath', 'save_top_k', 'monitor', 'mode', 'save_last', 'save_final'}
        self._validate_keys("CheckpointConfig", data.keys(), expected_keys)
        try:
            self.dirpath: str = str(data.get('dirpath', 'checkpoints'))
            self.save_top_k: int = int(data.get('save_top_k', 1)) # Default to saving best
            # Default monitor might change depending on the primary goal (e.g., combined loss or specific task loss)
            self.monitor: str = str(data.get('monitor', 'val/loss_combined')) # Monitor combined loss by default
            self.mode: str = str(data.get('mode', 'min'))
            self.save_last: bool = bool(data.get('save_last', True))
            self.save_final: bool = bool(data.get('save_final', True)) # Save final model state dict separately
        except (ValueError, TypeError) as e:
            print(f"Error parsing CheckpointConfig: {e}. Input data: {data}")
            raise

    def _validate_keys(self, class_name: str, provided_keys: set, expected_keys: set):
        """Warns about unexpected keys in the config dictionary."""
        provided_keys_set = set(provided_keys)
        unexpected_keys = provided_keys_set - expected_keys
        if unexpected_keys:
            warnings.warn(
                f"In {class_name}: Unexpected keys found in config dictionary and will be ignored: {unexpected_keys}",
                UserWarning
            )

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

class LoggingConfig:
    # --- No changes needed here for mixed training ---
    def __init__(self, data: Dict[str, Any]):
        expected_keys = {'project', 'entity'} # Add 'entity' if commonly used
        self._validate_keys("LoggingConfig", data.keys(), expected_keys)
        try:
            self.project: str = str(data.get('project', 'PolarBERT-Default-Project'))
            self.entity: Optional[str] = data.get('entity', None) # WandB entity
        except (ValueError, TypeError) as e:
            print(f"Error parsing LoggingConfig: {e}. Input data: {data}")
            raise

    def _validate_keys(self, class_name: str, provided_keys: set, expected_keys: set):
        """Warns about unexpected keys in the config dictionary."""
        provided_keys_set = set(provided_keys)
        unexpected_keys = provided_keys_set - expected_keys
        if unexpected_keys:
            warnings.warn(
                f"In {class_name}: Unexpected keys found in config dictionary and will be ignored: {unexpected_keys}",
                UserWarning
            )

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

class TrainingConfig:
    def __init__(self, data: Dict[str, Any]):
         # Define expected keys for this config section
        expected_keys = {
            # Training loop
            'max_epochs', 'logical_batch_size', 'val_check_interval', 'gpus',
            'precision', 'gradient_clip_val',
            # Task & Model Specific
            'task', 'directional_pooling_mode', 'freeze_backbone',
            'pretrained_checkpoint_path', # Allow setting initial checkpoint here
            # Loss Weights (NEW)
            'lambda_dom', # Original aux dom loss weight (maybe keep for backward compat?)
            'lambda_dom_kaggle', 'lambda_dom_prometheus', 'lambda_dir_prometheus',
            # Optimizer
            'optimizer', 'max_lr', 'adam_beta1', 'adam_beta2', 'adam_eps',
            'weight_decay', 'amsgrad',
             # Scheduler
            'lr_scheduler', 'warm_up_steps', 'pct_start', 'div_factor', 'final_div_factor',
            # Nested Configs
            'checkpoint', 'logging'
        }
        self._validate_keys("TrainingConfig", data.keys(), expected_keys)

        try:
            # --- Training loop ---
            self.max_epochs: int = int(data.get('max_epochs', 10))
            self.logical_batch_size: int = int(data.get('logical_batch_size', 2048))
            self.val_check_interval: Any = data.get('val_check_interval', 1.0) # Float (fraction) or Int (batches)
            self.gpus: Union[int, List[int], str] = data.get('gpus', 1)
            self.precision: str = str(data.get('precision', '16-mixed'))
            self.gradient_clip_val: Optional[float] = data.get('gradient_clip_val', 1.0)
            if self.gradient_clip_val is not None: self.gradient_clip_val = float(self.gradient_clip_val)

            # --- Fine-tuning / Multi-Task Specific ---
            self.task: str = str(data.get('task', 'direction'))
            self.directional_pooling_mode: str = str(data.get('directional_pooling_mode', 'cls')).lower()
            self.freeze_backbone: bool = bool(data.get('freeze_backbone', False))
            self.pretrained_checkpoint_path: Optional[str] = data.get('pretrained_checkpoint_path', None)

            # --- Loss Weights (NEW) ---
            self.lambda_dom: float = float(data.get('lambda_dom', 1.0)) # Aux DOM loss weight (maybe deprecated)
            self.lambda_dom_kaggle: float = float(data.get('lambda_dom_kaggle', 1.0))
            self.lambda_dom_prometheus: float = float(data.get('lambda_dom_prometheus', 1.0))
            self.lambda_dir_prometheus: float = float(data.get('lambda_dir_prometheus', 1.0))

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
            self.warm_up_steps: Optional[int] = data.get('warm_up_steps', None)
            if self.warm_up_steps is not None: self.warm_up_steps = int(self.warm_up_steps)
            self.pct_start: float = float(data.get('pct_start', 0.01))
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

    def _validate_keys(self, class_name: str, provided_keys: set, expected_keys: set):
        """Warns about unexpected keys in the config dictionary."""
        provided_keys_set = set(provided_keys)
        unexpected_keys = provided_keys_set - expected_keys
        if unexpected_keys:
            warnings.warn(
                f"In {class_name}: Unexpected keys found in config dictionary and will be ignored: {unexpected_keys}",
                UserWarning
            )

    def calculate_runtime_params(self, total_device_steps: int):
        """Calculates derived parameters based on device steps and batch sizes."""
        if total_device_steps <= 0:
            warnings.warn("total_device_steps is zero or negative. Step calculations might be incorrect.")
            total_device_steps = 1

        print("Calculating runtime training parameters...")
        if self.per_device_batch_size is None or self.gradient_accumulation_steps is None:
             raise RuntimeError("per_device_batch_size and gradient_accumulation_steps must be set before calling calculate_runtime_params.")

        self.effective_batch_size = self.gradient_accumulation_steps * self.per_device_batch_size * (self.gpus if isinstance(self.gpus, int) else 1)
        print(f"  Logical Batch Size: {self.logical_batch_size}")
        print(f"  Per Device Batch Size: {self.per_device_batch_size}")
        print(f"  Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        print(f"  Approx Effective Batch Size (GPUs={self.gpus}): {self.effective_batch_size}")

        self.total_steps = math.ceil(total_device_steps / self.gradient_accumulation_steps)
        print(f"  Total Device Steps (Batches * Epochs): {total_device_steps}")
        print(f"  Total Optimizer Steps: {self.total_steps}")

        if self.warm_up_steps is not None and self.warm_up_steps > 0:
            if self.total_steps > 0:
                calculated_pct_start = min(1.0, self.warm_up_steps / self.total_steps)
                if abs(calculated_pct_start - self.pct_start) > 1e-5:
                     print(f"  Overriding pct_start based on warm_up_steps: "
                           f"{self.pct_start:.4f} -> {calculated_pct_start:.4f} "
                           f"(Warmup: {self.warm_up_steps}, Total Optimizer Steps: {self.total_steps})")
                self.pct_start = calculated_pct_start
            else:
                 print("  Warning: Cannot calculate pct_start from warm_up_steps as total_steps is zero.")
        else:
            print(f"  Using configured pct_start: {self.pct_start:.4f}")

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d['checkpoint'] = self.checkpoint.to_dict()
        d['logging'] = self.logging.to_dict()
        return d

# --- Main Config Class ---
class PolarBertConfig:
    """Main configuration class orchestrating all sub-configurations."""
    def __init__(self, data_cfg: Dict, model_cfg: Dict, training_cfg: Dict):
        self.data = DataConfig(data_cfg)
        self.model = ModelConfig(model_cfg)
        self.training = TrainingConfig(training_cfg)
        # Store max_per_device_batch_size in training config for easy access during runtime calc
        # self.training.max_per_device_batch_size = self.data.max_per_device_batch_size # Added this line
        self._validate() # Call validation method on initialization

    @classmethod
    def from_yaml(cls, path: str) -> 'PolarBertConfig':
        """Loads configuration from a YAML file."""
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

        if 'embedding' not in cfg_dict.get('model', {}): warnings.warn("YAML model section missing 'embedding'.")
        if 'logging' not in cfg_dict.get('training', {}): warnings.warn("YAML training section missing 'logging'.")
        if 'checkpoint' not in cfg_dict.get('training', {}): warnings.warn("YAML training section missing 'checkpoint'.")

        return cls(
            data_cfg=cfg_dict.get('data', {}),
            model_cfg=cfg_dict.get('model', {}),
            training_cfg=cfg_dict.get('training', {})
        )

    def calculate_runtime_params(self, total_device_steps: int):
        """Calculates runtime parameters (steps, batches) after dataloader info is available."""
        self.training.calculate_runtime_params(total_device_steps)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the config object (including nested ones) to a dictionary."""
        return {
            'data': self.data.to_dict(),
            'model': self.model.to_dict(),
            'training': self.training.to_dict()
        }

    def save_yaml(self, path: str):
        """Saves the current configuration state to a YAML file."""
        print(f"Saving configuration to: {path}")
        target_dir = os.path.dirname(path)
        if target_dir: os.makedirs(target_dir, exist_ok=True)
        cfg_dict = self.to_dict()
        try:
            with open(path, 'w') as f:
                yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False, indent=2)
        except IOError as e:
            print(f"Error: Could not write config file to {path}: {e}")
            raise

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config_filename="config.yaml") -> 'PolarBertConfig':
        """Loads configuration from a YAML file stored with a checkpoint."""
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        config_path = os.path.join(os.path.dirname(checkpoint_path), config_filename)
        if not os.path.isfile(config_path):
             config_path_final = os.path.join(os.path.dirname(checkpoint_path), "final_config.yaml")
             if os.path.isfile(config_path_final): config_path = config_path_final
             else:
                raise FileNotFoundError(
                    f"Config file ('{config_filename}' or 'final_config.yaml') not found in {os.path.dirname(checkpoint_path)}"
                )
        print(f"Loading config associated with {os.path.basename(checkpoint_path)} from {os.path.basename(config_path)}")
        return cls.from_yaml(config_path)

    def _validate(self):
        """Performs configuration validation checks."""
        print("Validating configuration...")
        # --- Model Validation ---
        if self.model.embedding_dim % self.model.num_heads != 0:
            raise ValueError(f"model.embedding_dim ({self.model.embedding_dim}) must be divisible by model.num_heads ({self.model.num_heads})")
        if self.model.use_rope and self.model.use_positional_embedding:
            raise ValueError("Cannot use both RoPE and standard positional embeddings.")

        # --- Embedding Validation ---
        emb_cfg = self.model.embedding
        if not emb_cfg.embedding_projection and emb_cfg._sum_sub_dims != self.model.embedding_dim:
            raise ValueError(f"embedding_projection is False, but sum of sub-dims ({emb_cfg._sum_sub_dims}) != model.embedding_dim ({self.model.embedding_dim}).")

        # --- Training Validation ---
        train_cfg = self.training
        # Pooling mode validated in TrainingConfig init
        if isinstance(train_cfg.lr_scheduler, str) and train_cfg.lr_scheduler.lower() == 'onecycle':
            if not isinstance(train_cfg.final_div_factor, (int, float)) or train_cfg.final_div_factor <= 0:
                raise ValueError(f"training.final_div_factor must be > 0 for OneCycleLR, got {train_cfg.final_div_factor}")
            if train_cfg.pct_start < 0 or train_cfg.pct_start > 1:
                raise ValueError(f"training.pct_start must be in [0, 1], got {train_cfg.pct_start}")

        # --- Data Validation ---
        if not self.data.train_dir or not os.path.isdir(self.data.train_dir): warnings.warn(f"Kaggle train_dir '{self.data.train_dir}' not found.")
        if not self.data.val_dir or not os.path.isdir(self.data.val_dir): warnings.warn(f"Kaggle val_dir '{self.data.val_dir}' not found.")
        if not self.data.prometheus_dir or not os.path.isdir(self.data.prometheus_dir): warnings.warn(f"Prometheus dir '{self.data.prometheus_dir}' not found.")
        if self.data.prometheus_train_events is None or self.data.prometheus_val_events is None: warnings.warn("Prometheus event counts not set; splitting might fail.")

        print("Configuration validation passed (with potential warnings).")