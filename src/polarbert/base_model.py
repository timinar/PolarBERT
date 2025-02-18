import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
from polarbert.embedding import IceCubeEmbedding

class SimpleTransformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = IceCubeEmbedding(config, masking=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['model']['embedding_dim'],
            nhead=config['model']['num_heads'],
            dim_feedforward=config['model']['hidden_size'],
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['model']['num_layers'])
        num_doms = 5160
        self.unembedding = nn.Linear(config['model']['embedding_dim'], num_doms + 1)
        self.charge_prediction = nn.Linear(config['model']['embedding_dim'], 1)
        self.lambda_charge = config['model']['lambda_charge']

    def forward(self, x):
        embeddings, padding_mask, mask = self.embedding(x)
        output = self.transformer(embeddings)
        cls_embed = output[:, 0, :]
        charge = self.charge_prediction(cls_embed)
        logits = self.unembedding(output[:, 1:, :])
        return logits, mask, charge, padding_mask[:, 1:]
    
    def shared_step(self, batch):
        inp, y = batch
        logits, mask, charge_hat, padding_mask = self(inp)
        x, l = inp
        _, charge = y
        loss = self.masked_prediction_loss(logits, x[:, :, 3].long(), mask, padding_mask)
        charge_loss = F.mse_loss(charge_hat.squeeze(), torch.log10(charge))
        return loss, charge_loss

    def training_step(self, batch, batch_idx):
        loss, charge_loss = self.shared_step(batch)
        self.log('train/dom_loss', loss, prog_bar=True)
        self.log('train/charge_loss', charge_loss, prog_bar=True)
        full_loss = loss + self.lambda_charge * charge_loss
        self.log('train/full_loss', full_loss, prog_bar=True)
        return full_loss

    def validation_step(self, batch, batch_idx):
        loss, charge_loss = self.shared_step(batch)
        self.log('val/dom_loss', loss, prog_bar=True)
        self.log('val/charge_loss', charge_loss, prog_bar=True)
        full_loss = loss + self.lambda_charge * charge_loss
        self.log('val/full_loss', full_loss, prog_bar=True)
        return full_loss
    
    def masked_prediction_loss(self, logits, target_dom_ids, mask, padding_mask):
        mask = mask & ~padding_mask
        loss = F.cross_entropy(logits.flatten(0, 1), target_dom_ids.flatten(), reduction='none')
        loss = (loss * mask.flatten()).sum() / (mask.sum() + 1e-8)
        return loss
    
    def configure_optimizers(self):
        return _configure_optimizers(self.config, self.parameters())


def _configure_optimizers(config, parameters):

    if config['training']['lr_scheduler'] == 'constant':
        initial_lr = float(config['training']['initial_lr'])
    elif config['training']['lr_scheduler'] == 'onecycle':
        initial_lr = float(config['training']['max_lr']) / float(config['training']['div_factor'])
    else:
        raise ValueError(f"Unknown scheduler: {config['training']['lr_scheduler']}")
    
    optimizer = torch.optim.AdamW(
        parameters,
        lr=initial_lr,
        betas=(
            float(config['training'].get('adam_beta1', 0.9)),
            float(config['training'].get('adam_beta2', 0.999))
        ),
        eps=float(config['training'].get('adam_eps', 1e-8)),
        weight_decay=float(config['training']['weight_decay']),
        amsgrad=bool(config['training'].get('amsgrad', False))
    )

    if config['training']['lr_scheduler'] == 'constant':
        return optimizer
    elif config['training']['lr_scheduler'] == 'onecycle':
        # Use the pre-calculated total_steps from config
        total_steps = config['training']['total_steps']
        scheduler = OneCycleLR(
            optimizer,
            max_lr=float(config['training']['max_lr']),
            total_steps=total_steps,
            pct_start=float(config['training']['pct_start']),
            div_factor=float(config['training']['div_factor']),
            final_div_factor=float(config['training']['final_div_factor']),
            anneal_strategy='cos'
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    else:
        assert False
