import torch
import numpy as np
from typing import Any
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule, Trainer
from functools import partial
from pathlib import Path
import json

class CoordinateCheckingCallback(Callback):
    """
    Callback to log the typical activation magnitudes during training.
    This is used to test the implementation of the maximal update parameterization, by checking that the activations are independent of the model width.
    See https://blog.eleuther.ai/mutransfer/ for more details.
    """
    def __init__(self, checkpoint_dir: Path):
        super().__init__()
        self.coord_check_dict = dict()
        self.coord_check_staging_dict = dict()
        self.coord_check_handles = list()
        self.save_path = checkpoint_dir / 'coordinate_check.json'
        try:
            self.save_path.touch(exist_ok=True)
        except Exception as e:
            raise Exception(f"Error creating coordinate check file: {e}")
        
    def _stage_to_dict(self, key, value):
        # Collect data at every step (including gradient accumulation steps)
        if key not in self.coord_check_staging_dict:
            self.coord_check_staging_dict[key] = []
        self.coord_check_staging_dict[key].append(value)

    def _append_average_values_to_dict(self):
        # Average over mini-batches when using gradient accumulation
        if not self.coord_check_staging_dict:
            return
        for key, values in self.coord_check_staging_dict.items():
            assert values, f"No values collected for key '{key}' - this likely indicates an internal error in `CoordinateCheckingCallback`."
            if key not in self.coord_check_dict:
                self.coord_check_dict[key] = []
            self.coord_check_dict[key].append(np.mean(values))
        self.coord_check_staging_dict.clear()

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Register hooks once at the start of training."""
        # Prevent double-registration if this method is called multiple times
        if self.coord_check_handles:
            return

        def standard_hook(module, input, output, key):
            with torch.no_grad():
                self._stage_to_dict(key, output.abs().mean().item())

        def embedding_hook(module, input, output):
            with torch.no_grad():
                full_embedding, padding_mask, mask = output
                cls = full_embedding[:,0,:]
                combined_embeds = full_embedding[:,1:,:]
                unmasked_embeds = combined_embeds[~mask]
                masked_embeds = combined_embeds[mask]
                self._stage_to_dict('embedding.cls_embedding', cls.abs().mean().item())
                self._stage_to_dict('embedding.combined_embeds[~mask]', unmasked_embeds.abs().mean().item())
                self._stage_to_dict('embedding.combined_embeds[mask]', masked_embeds.abs().mean().item())

        for module_name, module in pl_module.named_modules():
            if module_name in ['embedding.dom_embedding', 'embedding.position_embedding', 'embedding.features_embedding', 'charge_prediction', 'unembedding'] or \
               module_name.endswith('.attention') or module_name.endswith('.feed_forward'):
                self.coord_check_handles.append(module.register_forward_hook(partial(standard_hook, key=module_name)))
            # Special treatment to separate the CLS and mask token embeddings
            elif module_name == 'embedding':
                self.coord_check_handles.append(module.register_forward_hook(embedding_hook))

    def on_before_optimizer_step(self, trainer: Trainer, pl_module: LightningModule, optimizer: Any) -> None:
        """Average and log the collected values when optimizer steps."""
        self._append_average_values_to_dict()

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Log any remaining staged values that haven't been logged yet
        # This handles the case where training ends mid-accumulation cycle
        self._append_average_values_to_dict()

        # Clean up hooks
        for handle in self.coord_check_handles:
            handle.remove()
        self.coord_check_handles.clear()

        # Save results
        with open(self.save_path, 'w') as f:
            json.dump(self.coord_check_dict, f, indent=4)
