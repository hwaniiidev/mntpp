import torch
import numpy as np


_positional_encoding_cache = {}


def positional_encoding(src):
    # The output depends only on src's shape/dtype/device, so compute once and reuse
    # (avoids per-element GPU writes on every forward pass)
    key = (tuple(src.shape), src.dtype, src.device)
    if key not in _positional_encoding_cache:
        _positional_encoding_cache[key] = _compute_positional_encoding(src)
    return _positional_encoding_cache[key]


def _compute_positional_encoding(src):
    pos_encoding = torch.zeros_like(src)
    seq_len = pos_encoding.shape[0]
    d_model = pos_encoding.shape[1]

    for i in range(d_model):
        for pos in range(seq_len):
            if i % 2 == 0:
                pos_encoding[pos, i] = np.sin(pos / 100 ** (2 * i / d_model))
            else:
                pos_encoding[pos, i] = np.cos(pos / 100 ** (2 * i / d_model))
    return pos_encoding.float()


def second_positional_encoding(src):
    batch_size, total_seq_frag_num, feature_dim = src.shape
    pos_encoding = torch.zeros(
        batch_size, total_seq_frag_num, feature_dim, device=src.device
    )

    for i in range(feature_dim):
        for pos in range(total_seq_frag_num):
            if i % 2 == 0:
                pos_encoding[:, pos, i] = torch.sin(
                    torch.tensor(pos, dtype=torch.float32, device=src.device)
                    / 100 ** (2 * i / feature_dim)
                )
            else:
                pos_encoding[:, pos, i] = torch.cos(
                    torch.tensor(pos, dtype=torch.float32, device=src.device)
                    / 100 ** (2 * i / feature_dim)
                )
    return pos_encoding
