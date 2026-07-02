import torch
import torch.nn as nn

from config_size4_projection import *
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

    def forward(self, X):
        # print("forward_X : ", X)

        feed_time = X[:, :, :1]
        feed_size = X[:, :, 1]

        X_time = self.lin0(feed_time.float())
        X_size = self.emb_size(feed_size.int())

        X_cat = torch.cat((X_time, X_size), 2)
        X_cat = X_cat.float()

        src = X_cat + positional_encoding(X_cat).to(device)

        src = src.float()

        X_cat_seqnet = self.encoder_layer(src)
        # print("X_cat_seqnet : ", X_cat_seqnet)
        # print("X_cat_seqnet[:, -1, :] :", X_cat_seqnet[:, -1, :])
        x_relu = self.lin_relu(X_cat_seqnet[:, -1, :])

        # print("forward_X_relu : ", x_relu)

        return x_relu


class ForecasterNTPP(nn.Module):
    def __init__(self):
        super(ForecasterNTPP, self).__init__()
        self.lin_time = nn.Linear(input_features_len, 1)
        self.lin_size = nn.Linear(input_features_len + 1, 4)

        self.NN_time = nn.ModuleList()
        self.NN_size = nn.ModuleList()

        for num_layer_time in range(1):
            self.NN_time.append(nn.Linear(input_features_len, input_features_len))

        for num_layer_size in range(1):
            self.NN_size.append(
                nn.Linear(input_features_len + 1, input_features_len + 1)
            )

    def forward(self, x_relu):
        model_time = x_relu
        for layer in self.NN_time[:]:
            model_time = layer(model_time)
        model_time = self.lin_time(model_time)

        features_size = torch.cat((model_time, x_relu), 1)
        model_size = features_size
        for layer in self.NN_size[:]:
            model_size = layer(model_size)
        model_size = self.lin_size(model_size)

        out = torch.cat((model_time, model_size), 1)

        return out
