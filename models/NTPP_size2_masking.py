import torch
import torch.nn as nn

from config_size2 import *
from utils.encoding_utils import *


class EncoderNTPP(nn.Module):
    def __init__(self):
        super(EncoderNTPP, self).__init__()
        self.lin0 = nn.Linear(pkt_time_lin_in, pkt_time_lin_out, bias=True)
        self.emb_size = nn.Embedding(
            pkt_size_emb_in, pkt_size_emb_out, scale_grad_by_freq=scale_grad_by_freq
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_features_len,
            nhead=multihead_attention,
            batch_first=True,
            dim_feedforward=hidden_dim,
        ).to(device)

        self.lin_relu = nn.Linear(input_features_len, input_features_len)

    def forward(self, X, lengths):
        feed_time = X[:, :, :1]
        feed_size = X[:, :, 1]

        X_time = self.lin0(feed_time.float())
        X_size = self.emb_size(feed_size.int())

        X_cat = torch.cat((X_time, X_size), 2)
        X_cat = X_cat.float()

        src = X_cat + positional_encoding(X_cat).to(device)
        src = src.float()

        max_len = src.size(1)
        mask = torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]

        X_cat_seqnet = self.encoder_layer(src, src_key_padding_mask=mask)

        last_indices = (
            (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, X_cat_seqnet.size(2))
        )
        last_hidden = X_cat_seqnet.gather(1, last_indices).squeeze(1)

        x_relu = self.lin_relu(last_hidden)

        return x_relu
