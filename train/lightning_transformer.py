import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from transformer_model import FridayTransformer
from dataset import FridayDataset


class LightningTransformer(pl.LightningModule):
    def __init__(
            self,
            embed_dim,
            vocab_size,
            num_layers,
            n_heads,
            learning_rate,
            sp_model,
            max_len,
            train_data=None,
            val_data=None
    ):
        super().__init__()
        self.dataset = None
        self.val_dataset = None
        self.model = FridayTransformer(embed_dim=embed_dim,
                                       src_vocab_size=vocab_size,
                                       target_vocab_size=vocab_size,
                                       num_layers=num_layers,
                                       n_heads=n_heads)
        self.criterion = nn.CrossEntropyLoss(ignore_index=sp_model.pad_id())
        self.learning_rate = learning_rate
        self.sp_model = sp_model
        self.max_len = max_len
        self.inputs = None
        self.targets = None
        self.val_inputs = None
        self.val_targets = None
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = 5

    def forward(self, src, trg_input):
        return self.model(src, trg_input)

    def training_step(self, batch, batch_idx):
        src, trg = batch
        # Move tensors to the same device as the model
        src = src.to(self.device)
        trg = trg.to(self.device)

        trg_input = trg[:, :-1]  # Remove last token
        prediction = self(src, trg_input)  # Forward pass
        prediction = prediction.reshape(-1, prediction.shape[-1])  # Reshape for loss
        trg = trg[:, 1:].reshape(-1)  # Remove first token for targets

        loss = self.criterion(prediction, trg)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, trg = batch
        # Move tensors to the same device as the model
        src = src.to(self.device)
        trg = trg.to(self.device)

        trg_input = trg[:, :-1]  # Remove last token
        prediction = self(src, trg_input)  # Forward pass
        prediction = prediction.reshape(-1, prediction.shape[-1])  # Reshape for loss
        trg = trg[:, 1:].reshape(-1)  # Remove first token for targets

        loss = self.criterion(prediction, trg)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def prepare_data(self):
        if self.train_data is None:
            raise ValueError("Train Data not provided.")
        inputs = [pair[0] for pair in self.train_data]
        targets = [pair[1] for pair in self.train_data]
        # Shuffle data to ensure randomness
        self.inputs, self.targets = shuffle(inputs, targets)
        if self.val_data is None:
            raise ValueError("Validation Data not provided.")
        val_inputs = [pair[0] for pair in self.val_data]
        val_targets = [pair[1] for pair in self.val_data]
        self.val_inputs, self.val_targets = shuffle(val_inputs, val_targets)


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.dataset = FridayDataset(self.inputs, self.targets, self.max_len, self.sp_model)
            self.val_dataset = FridayDataset(self.val_inputs, self.val_targets, self.max_len, self.sp_model)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=7)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=7)
