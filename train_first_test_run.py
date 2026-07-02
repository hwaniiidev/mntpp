import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from datetime import datetime

import utils.utils as utils
import utils.data_utils_bin4 as data_utils
from config_size4 import *
from models.NTPP_size4 import EncoderNTPP, ForecasterNTPP

max_len = 9   # 이 CSV는 pktTime0~8 (9개)만 존재
epochs = 2    # 테스트용
batch_size, num_workers = 10, 0
window_size = 4

Alpha_time_weight = 8
HTTP_time_weight = 3
Multi_time_weight = 6
Unlabeled_time_weight = 5

csv_path = "0901_30_K_5-10_T_1.0s.csv"
n_streams = 10000

label_dict = yaml.safe_load(open("config_label.yaml", "r"))

valid_pkt_time_columns = [f"pktTime{i}" for i in range(max_len)]
valid_pkt_size_columns = [f"pktSize{i}" for i in range(max_len)]

train, _, _ = data_utils.define_data(csv_path, max_len, n_streams, label_dict)


def split_by_label(df):
    Alpha_sequences = df[df["taxonomy label"] == 0].reset_index(drop=True)
    HTTP_sequences = df[df["taxonomy label"] == 1].reset_index(drop=True)
    Multi_sequences = df[df["taxonomy label"] == 2].reset_index(drop=True)
    Unlabeled_sequences = df[df["taxonomy label"] == 3].reset_index(drop=True)
    return Alpha_sequences, HTTP_sequences, Multi_sequences, Unlabeled_sequences


(
    Alpha_train_sequences,
    HTTP_train_sequences,
    Multi_train_sequences,
    Unlabeled_train_sequences,
) = split_by_label(train)

print(f"Alpha: {len(Alpha_train_sequences)}, HTTP: {len(HTTP_train_sequences)}, "
      f"Multi: {len(Multi_train_sequences)}, Unlabeled: {len(Unlabeled_train_sequences)}")


def concat_pktTime_pktSize(df):
    concat_time = []
    concat_size = []
    seq_id = []

    for i in range(len(df)):
        valid_len = int(df["len"].iloc[i])
        valid_len = min(valid_len, max_len)  # pktTime 컬럼 수 초과 방지
        pkt_time_values = df[valid_pkt_time_columns].iloc[i, :valid_len].values
        pkt_size_values = df[valid_pkt_size_columns].iloc[i, :valid_len].values

        concat_time.extend(pkt_time_values)
        concat_size.extend(pkt_size_values)
        seq_id.extend([i] * valid_len)

    result_df = pd.DataFrame(
        {
            "seqId": seq_id,
            "pktTime": concat_time,
            "pktSize": concat_size,
        }
    )
    return result_df


Alpha_train_cat = concat_pktTime_pktSize(Alpha_train_sequences)
HTTP_train_cat = concat_pktTime_pktSize(HTTP_train_sequences)
Multi_train_cat = concat_pktTime_pktSize(Multi_train_sequences)

target_vars = ["pktTime", "pktSize"]


def valid_slice_flag(df, window_size):
    df["valid_slice_flag"] = True
    for i in range(len(df) - 1):
        if df.seqId[i] != df.seqId[i + 1]:
            df.loc[i + 1 - window_size : i, ["valid_slice_flag"]] = False
    df.loc[len(df) - window_size : len(df), ["valid_slice_flag"]] = False
    return df


Alpha_train_cat = valid_slice_flag(Alpha_train_cat, window_size)
HTTP_train_cat = valid_slice_flag(HTTP_train_cat, window_size)
Multi_train_cat = valid_slice_flag(Multi_train_cat, window_size)


def encode(df):
    df_tensor_time = torch.from_numpy(df["pktTime"].values)
    df_tensor_time = df_tensor_time.view(len(df_tensor_time), 1)

    df_tensor_size = torch.from_numpy(df["pktSize"].values)
    df_tensor_size = df_tensor_size.view(len(df_tensor_size), 1)

    encode_df = torch.cat((df_tensor_time, df_tensor_size), 1)
    return encode_df


encode_Alpha_train = encode(Alpha_train_cat)
encode_HTTP_train = encode(HTTP_train_cat)
encode_Multi_train = encode(Multi_train_cat)


class train_data:
    def __init__(self, train, encode_train):
        self.idx = np.repeat(True, len(train))
        self.valid_slice_idxn = np.where(
            np.logical_and(self.idx, train["valid_slice_flag"])
        )[0]
        self.train = train
        self.encode_train = encode_train

    def __len__(self):
        return int(np.sum(self.train.loc[self.idx, "valid_slice_flag"]))

    def __getitem__(self, i):
        j = self.valid_slice_idxn[i]
        x = self.encode_train[j : j + window_size]
        y = self.train.iloc[j + window_size].loc[target_vars]
        y = torch.from_numpy(y.to_numpy(dtype="float64"))
        return x, y


def cost_function(y, y_head, time_weight, size_weight):
    y_time = y[:, 0].float()
    y_size = y[:, 1].long()
    y_head_time = y_head[:, 0].float()
    y_head_size = y_head[:, 1:5]

    CEL_size = nn.CrossEntropyLoss(reduction="none")
    Yhat_CEL_size = torch.mean(CEL_size(y_head_size, y_size))
    Yhat_RMSE_time = torch.mean((y_time - y_head_time) ** 2) ** 0.5

    Yhat_CEL_size = Yhat_CEL_size * size_weight
    Yhat_RMSE_time = Yhat_RMSE_time * time_weight
    Loss = Yhat_CEL_size + Yhat_RMSE_time
    return Loss, Yhat_RMSE_time, Yhat_CEL_size


Alpha_train_dataset = train_data(Alpha_train_cat, encode_Alpha_train)
HTTP_train_dataset = train_data(HTTP_train_cat, encode_HTTP_train)
Multi_train_dataset = train_data(Multi_train_cat, encode_Multi_train)

Alpha_train_loader = DataLoader(Alpha_train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, drop_last=True)
HTTP_train_loader = DataLoader(HTTP_train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, drop_last=True)
Multi_train_loader = DataLoader(Multi_train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, drop_last=True)


def model_epoch(dataloader, train_dataloader, encoder, forecaster, optimizer, scheduler, epochtype, time_weight):
    if epochtype == "train":
        encoder.train()
        forecaster.train()
    else:
        encoder.eval()
        forecaster.eval()

    size = len(dataloader.dataset)
    loss_rollingmean, lossRMSE_rollingmean, lossCEL_rollingmean = 0.0, 0.0, 0.0

    for batch, (X, Y) in enumerate(dataloader):
        batch_total = len(train_dataloader)
        X, Y = X.to(device), Y.to(device)

        encoded_X = encoder(X)
        pred = forecaster(encoded_X)

        Loss, RMSE_time, CEL_size = cost_function(Y, pred, time_weight, 1)
        loss_rollingmean = loss_rollingmean + (Loss - loss_rollingmean) / (1 + batch)
        lossRMSE_rollingmean = lossRMSE_rollingmean + (RMSE_time - lossRMSE_rollingmean) / (1 + batch)
        lossCEL_rollingmean = lossCEL_rollingmean + (CEL_size - lossCEL_rollingmean) / (1 + batch)

        if epochtype == "train":
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

        if batch % 500 == 0:
            print(f"  batch {batch}/{batch_total} | loss: {Loss.item():.4f} | RMSE: {RMSE_time:.4f} | CE: {CEL_size:.4f}")

    return (
        loss_rollingmean.detach().cpu().numpy().item(),
        lossRMSE_rollingmean.detach().cpu().numpy().item(),
        lossCEL_rollingmean.detach().cpu().numpy().item(),
    )


def model_train(epochs, train_loader, time_weight, label):
    torch.cuda.empty_cache()
    import gc; gc.collect()

    encoder = EncoderNTPP().to(device)
    forecaster = ForecasterNTPP().to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(forecaster.parameters()),
        lr=0.001, eps=1e-16,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3)

    best_train_loss = float("inf")
    best_encoding_state_dict = None

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        import gc; gc.collect()
        print(f"\n[{label}] Epoch {epoch+1}/{epochs}")
        train_loss = model_epoch(train_loader, train_loader, encoder, forecaster, optimizer, scheduler, "train", time_weight)
        print(f"  => total loss: {train_loss[0]:.4f}, RMSE: {train_loss[1]:.4f}, CE: {train_loss[2]:.4f}")

        if train_loss[0] < best_train_loss:
            best_train_loss = train_loss[0]
            best_encoding_state_dict = encoder.state_dict()

        if optimizer.param_groups[0]["lr"] < 1e-7:
            break

    return best_encoding_state_dict, encoder


if __name__ == "__main__":
    utils.seed_everything(0)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("\n=== Training Alpha encoder ===")
    Alpha_params, Alpha_encoder = model_train(epochs, Alpha_train_loader, Alpha_time_weight, "Alpha")

    print("\n=== Training HTTP encoder ===")
    HTTP_params, HTTP_encoder = model_train(epochs, HTTP_train_loader, HTTP_time_weight, "HTTP")

    print("\n=== Training Multi encoder ===")
    Multi_params, Multi_encoder = model_train(epochs, Multi_train_loader, Multi_time_weight, "Multi")

    # Unlabeled: 이 데이터셋에 없으므로 스킵
    print("\n=== Unlabeled: 데이터 없음, 스킵 ===")

    import os
    os.makedirs("first_diff_weights", exist_ok=True)
    torch.save(Alpha_params, "./first_diff_weights/Alpha_model.pt")
    torch.save(HTTP_params, "./first_diff_weights/HTTP_model.pt")
    torch.save(Multi_params, "./first_diff_weights/Multi_model.pt")
    print("\n=== 저장 완료: first_diff_weights/ ===")
