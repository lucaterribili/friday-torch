import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformer_model import FridayTransformer


class LightningTransformer(pl.LightningModule):
    def __init__(self, embed_dim, vocab_size, num_layers, n_heads, learning_rate, sp_model):
        super().__init__()
        self.model = FridayTransformer(embed_dim=embed_dim,
                                 src_vocab_size=vocab_size,
                                 target_vocab_size=vocab_size,
                                 num_layers=num_layers,
                                 n_heads=n_heads)
        self.criterion = nn.CrossEntropyLoss(ignore_index=sp_model.pad_id())
        self.learning_rate = learning_rate

    def forward(self, src, trg_input):
        return self.model(src, trg_input)

    def training_step(self, batch, batch_idx):
        src, trg = batch
        trg_input = trg[:, :-1]  # Remove last token
        prediction = self(src, trg_input)  # Forward pass
        prediction = prediction.reshape(-1, prediction.shape[-1])  # Reshape for loss
        trg = trg[:, 1:].reshape(-1)  # Remove first token for targets
        loss = self.criterion(prediction, trg)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)