import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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