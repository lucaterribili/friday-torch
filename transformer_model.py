import torch.nn.functional as F
import torch.nn as nn
import torch
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.W_out = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # q,k,v are of shape (batch_size,n_heads,seq_len,single_head_dim)
        d_k = q.size(-1)

        # score is of shape (batch_size,n_heads,seq_len,seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        # since masking is optional
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)
        attention_weights = torch.matmul(scores, v)
        # attention_weights is of shape (batch_size,n_heads,seq_len,single_head_dim)
        return attention_weights

    def forward(self, q, k, v, mask=None):
        """
            Q |
            K | -> scaled_dot_product_attention -> concat -> linear
            V |

        """
        # q,k,v are of shape (batch_size,seq_length,d_model)
        batch_size = q.size(0)
        # calculating linear projections
        # reshaping to (batch_size,seq_len,n_heads,single_head_dim) -> transpose to (batch_size,n_heads,seq_len,single_head_dim)
        q = self.W_Q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_K(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_V(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # attention
        attention = self.scaled_dot_product_attention(q, k, v, mask)
        batch_size, _, seq_length, d_k = attention.size()
        output = self.W_out(attention.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1) -> None:
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, input_vocab_size, max_len=512, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.positional_encoding = nn.Parameter(self._get_positional_encoding(max_len, d_model), requires_grad=False)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def _get_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x, mask=None):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, output_vocab_size, max_len=512, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_vocab_size, d_model)
        self.positional_encoding = nn.Parameter(self._get_positional_encoding(max_len, d_model), requires_grad=False)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, output_vocab_size)

    def _get_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        x = self.fc_out(x)
        return x


class FridayTransformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size, num_layers=6, d_ff=2048, n_heads=8) -> None:
        super(FridayTransformer, self).__init__()
        self.target_vocab_size = target_vocab_size
        self.encoder = Encoder(embed_dim, n_heads, d_ff, num_layers, src_vocab_size)
        self.decoder = Decoder(embed_dim, n_heads, d_ff, num_layers, target_vocab_size)
        self.num_heads = n_heads

    def generate_mask(self, src, trg):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        batch_size, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(batch_size, 1, trg_len, trg_len)
        return src_mask, trg_mask

    def forward(self, src, trg):
        src_mask, trg_mask = self.generate_mask(src, trg)
        enc_out = self.encoder(src, src_mask)

        outputs = self.decoder(trg, enc_out, src_mask, trg_mask)
        return outputs