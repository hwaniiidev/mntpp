import torch

attention_layer = 1
hidden_dim = 1024
multihead_attention = 1
scale_grad_by_freq = True

pkt_size = [0, 1, 2, 3]
pkt_time_lin_in = 1
pkt_time_lin_out = 1
pkt_size_emb_in = len(pkt_size)
pkt_size_emb_out = len(pkt_size)
input_features_len = pkt_time_lin_out + pkt_size_emb_out
hist_dim = 1 + 4

device = "cuda:0" if torch.cuda.is_available() else "cpu"
