!pip install pytorch-lightning

import os
import glob
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sentencepiece as spm
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from sklearn.utils import shuffle
############################## DATASET

class SaturdayDataset(Dataset):
    def __init__(self, source_sentences, target_sentences, max_len, tokenizer):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.filter_invalid_pairs()

    def __len__(self):
        return len(self.source_sentences)

    def pad_sequence(self, original_source, original_target):
        # Encode the source and target sequences
        source = original_source[:]
        target = original_target[:]
        # Insert BOS token at the beginning of the target sequence
        target.insert(0, self.tokenizer.bos_id())
        target.append(self.tokenizer.eos_id())  # Add EOS token at the end

        # Calculate lengths
        source_len = len(source)
        target_len = len(target)

        # Padding for target sequence (add padding after EOS)
        remaining_padding_for_target = self.max_len - target_len
        if remaining_padding_for_target > 0:
            target = target + [self.tokenizer.pad_id()] * remaining_padding_for_target

        # Padding for source sequence
        remaining_padding_for_source = self.max_len - source_len
        if remaining_padding_for_source > 0:
            source = source + [self.tokenizer.pad_id()] * remaining_padding_for_source

        return source, target

    def filter_invalid_pairs(self):
        valid_sources = []
        valid_targets = []
        for src, trg in zip(self.source_sentences, self.target_sentences):
            padded_src, padded_trg = self.pad_sequence(src, trg)
            # Verifica che entrambi siano validi
            if len(padded_src) <= self.max_len and len(padded_trg) <= self.max_len:
                valid_sources.append(src)
                valid_targets.append(trg)

        # Aggiorna le liste con i dati validi
        self.source_sentences = valid_sources
        self.target_sentences = valid_targets

    def __getitem__(self, idx):
        src = self.source_sentences[idx]
        trg = self.target_sentences[idx]
        # Converti le frasi in indici e aggiungi il padding
        src_indexes, trg_indexes = self.pad_sequence(src, trg)

        return torch.tensor(src_indexes), torch.tensor(trg_indexes)
################################# MODEL
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SaturdayTransformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size, num_layers=6, d_ff=2048, n_heads=8, dropout=0.1):
        super(SaturdayTransformer, self).__init__()
        self.target_vocab_size = target_vocab_size

        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=n_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )

        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_padding_mask = (src == 0).to(dtype=torch.bool)  # Tipo bool
        tgt_padding_mask = (tgt == 0).to(dtype=torch.bool)  # Tipo bool

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1),
            device=tgt.device
        ).to(dtype=torch.bool)  # Uniforma il tipo a bool

        src_emb = self.pos_encoder(self.src_embedding(src))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt))

        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        return self.fc_out(output)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        # Creare src_mask direttamente qui
        src_padding_mask = (src == 0).to(dtype=torch.bool)
        src_emb = self.pos_encoder(self.src_embedding(src))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Usare il metodo predefinito per la tgt_mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1),
            device=tgt.device
        )
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)

    def generate(self, src: torch.Tensor, start_token: int, max_len: int = 50,
                 temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            # Qui usiamo il metodo encode aggiornato che già gestisce la maschera correttamente
            memory = self.encode(src)
            batch_size = src.size(0)
            tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=src.device)

            for _ in range(max_len - 1):
                # Qui usiamo il metodo decode aggiornato
                output = self.decode(tgt, memory)
                logits = self.fc_out(output[:, -1:, :]) / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs.squeeze(1), 1)
                tgt = torch.cat([tgt, next_token], dim=1)

            return tgt

################################# LIGHTING
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
            val_data=None,
            batch_size=50
    ):
        super().__init__()
        self.dataset = None
        self.val_dataset = None
        self.model = SaturdayTransformer(embed_dim=embed_dim,
                                         src_vocab_size=vocab_size,
                                         target_vocab_size=vocab_size,
                                         num_layers=num_layers,
                                         n_heads=n_heads
                                         )
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
        self.batch_size = batch_size
        self.tokenizer = sp_model
        self.pad_id = sp_model.pad_id()

    def forward(self, src, trg_input):
        return self.model(src, trg_input)

    def training_step(self, batch, batch_idx):
        src, trg = batch
        # src e trg sono ora di forma [batch_size, sequence_length]

        # Move tensors to the same device as the model
        src = src.to(self.device)
        trg = trg.to(self.device)
        tgt_input = trg[:, :-1]  # Prendiamo tutto tranne l'ultimo token (l'input per il decoder)


        # Passiamo i dati nel modello
        logits = self(src, tgt_input)  # Forward pass

        # L'output del target è senza il primo token
        tgt_out = trg[:, 1:]

        # Calcoliamo la loss
        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, trg = batch
        # src e trg sono ora di forma [batch_size, sequence_length]

        # Move tensors to the same device as the model
        src = src.to(self.device)
        trg = trg.to(self.device)
        tgt_input = trg[:, :-1]  # Prendiamo tutto tranne l'ultimo token (l'input per il decoder)


        # Passiamo i dati nel modello
        logits = self(src, tgt_input)  # Forward pass

        # L'output del target è senza il primo token
        tgt_out = trg[:, 1:]

        # Calcoliamo la loss
        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
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
            self.dataset = SaturdayDataset(self.inputs, self.targets, self.max_len, self.sp_model)
            self.val_dataset = SaturdayDataset(self.val_inputs, self.val_targets, self.max_len, self.sp_model)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=5)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=5)
########################### INIT
tokenizer_path = '/content/drive/MyDrive/models/tokenizer/model.model'
sp_model = spm.SentencePieceProcessor()
sp_model.Load(tokenizer_path)

# Load inputs and targets
train_data = np.load('/content/drive/MyDrive/dataset/transformer/dataset.npy', allow_pickle=True)
validation_data = np.load('/content/drive/MyDrive/dataset/transformer/dataset_validation.npy', allow_pickle=True)

# Parametri del modello
embed_dim = 512
vocab_size = sp_model.vocab_size()
num_layers = 3
n_heads = 8
learning_rate = 0.0001
max_len = 350

model = LightningTransformer(
    embed_dim=embed_dim,
    vocab_size=vocab_size,
    num_layers=num_layers,
    n_heads=n_heads,
    learning_rate=learning_rate,
    sp_model=sp_model,
    max_len=max_len,
    train_data=train_data,
    val_data=validation_data,
    batch_size=40
)

model_dir = os.path.join("/content/drive/MyDrive/models/transformer/", "discussion.ckpt")

if os.path.exists(model_dir):  # Controllo corretto
    try:
        pretrained = torch.load(model_dir, map_location="cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(pretrained["state_dict"], strict=False)
        print("Il modello è stato caricato correttamente.")
    except Exception as e:
        print(f"Errore nel caricare il modello: {e}")
else:
    print("Il file del modello non esiste nella posizione specificata.")

checkpoint_dir = "/content/drive/MyDrive/models/transformer/checkpoints/"

# Configurazione dei callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",  # Monitorare la perdita
    dirpath=checkpoint_dir,
    filename="transformer-{epoch:02d}-{val_loss:.2f}",  # Nome file
    save_top_k=10,  # Salva tutti i checkpoint
    every_n_epochs=1  # Salvataggio ad ogni epoca
)

early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    patience=10,
    verbose=True,
    mode="min"
)

trainer = pl.Trainer(
    max_epochs=10,
    accelerator='gpu',
    callbacks=[checkpoint_callback, early_stopping_callback]
)

# Trova tutti i file .ckpt nella cartella
checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))

if checkpoint_files:
    # Ordina i file per data di modifica (il più recente per ultimo)
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    print(f"Checkpoint trovato: Riprendo l'allenamento da {latest_checkpoint}")
    trainer.fit(model, ckpt_path=latest_checkpoint)
else:
    print("Checkpoint non trovato: Parto da zero.")
    trainer.fit(model)