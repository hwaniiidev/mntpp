import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import yaml
from datetime import datetime

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--week", type=str, default="week3_may2026")
_parser.add_argument("--condition", type=str, default="w9_ws4")
_parser.add_argument("--window_size", type=int, default=4)
_parser.add_argument("--alpha_w", type=float, default=9)
_parser.add_argument("--http_w", type=float, default=5)
_parser.add_argument("--multi_w", type=float, default=7)
_parser.add_argument("--unlabeled_w", type=float, default=11)
_parser.add_argument("--csv_path", type=str, default="all_combined_T1.0_kmax10.csv")
_parser.add_argument("--max_len", type=int, default=9)
_args, _ = _parser.parse_known_args()

WEEK = _args.week
CONDITION = _args.condition

import utils.utils as utils
import utils.data_utils_bin4 as data_utils
from config_size4 import *
from models.NTPP_size4 import EncoderNTPP, ForecasterNTPP

max_len = _args.max_len
epochs = 1000
batch_size, num_workers = 10, 0
window_size = _args.window_size

Alpha_time_weight = _args.alpha_w
HTTP_time_weight = _args.http_w
Multi_time_weight = _args.multi_w
Unlabeled_time_weight = _args.unlabeled_w

csv_path = _args.csv_path
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


def concat_pktTime_pktSize(df):
    concat_time = []
    concat_size = []
    seq_id = []

    for i in range(len(df)):
        valid_len = min(int(df["len"].iloc[i]), max_len)
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
Unlabeled_train_cat = concat_pktTime_pktSize(Unlabeled_train_sequences)


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
Unlabeled_train_cat = valid_slice_flag(Unlabeled_train_cat, window_size)

# Alpha_train_cat.to_csv("Alpha_train_cat_valid_slices.csv", index=False)


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
encode_Unlabeled_train = encode(Unlabeled_train_cat)


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
    # Loss = Yhat_RMSE_time * time_weight + Yhat_CEL_size * size_weight

    Yhat_CEL_size = Yhat_CEL_size * size_weight
    Yhat_RMSE_time = Yhat_RMSE_time * time_weight
    Loss = Yhat_CEL_size + Yhat_RMSE_time
    return Loss, Yhat_RMSE_time, Yhat_CEL_size


Alpha_train_dataset = train_data(Alpha_train_cat, encode_Alpha_train)
HTTP_train_dataset = train_data(HTTP_train_cat, encode_HTTP_train)
Multi_train_dataset = train_data(Multi_train_cat, encode_Multi_train)
Unlabeled_train_dataset = train_data(Unlabeled_train_cat, encode_Unlabeled_train)

Alpha_train_loader = DataLoader(
    Alpha_train_dataset,
    shuffle=True,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=True,
)


HTTP_train_loader = DataLoader(
    HTTP_train_dataset,
    shuffle=True,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=True,
)


Multi_train_loader = DataLoader(
    Multi_train_dataset,
    shuffle=True,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=True,
)


Unlabeled_train_loader = DataLoader(
    Unlabeled_train_dataset,
    shuffle=True,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=True,
)


def model_epoch(
    dataloader,
    train_dataloader,
    encoder,
    forecaster,
    optimizer,
    scheduler,
    epochtype,
    time_weight,
):
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
        # print("batch",batch+1,"/", batch_total)
        X, Y = X.to(device), Y.to(device)

        encoded_X = encoder(X)
        pred = forecaster(encoded_X)

        Loss, RMSE_time, CEL_size = cost_function(Y, pred, time_weight, 1)
        loss_rollingmean = loss_rollingmean + (Loss - loss_rollingmean) / (1 + batch)
        lossRMSE_rollingmean = lossRMSE_rollingmean + (
            RMSE_time - lossRMSE_rollingmean
        ) / (1 + batch)
        lossCEL_rollingmean = lossCEL_rollingmean + (CEL_size - lossCEL_rollingmean) / (
            1 + batch
        )

        if epochtype == "train":
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

        if batch % 500 == 0:
            print("batch", batch, "/", batch_total)
            loss = Loss
            loss, current = loss.item(), batch * X.shape[0]
            print(
                f"loss: {loss:>7f}, ln(1+loss): {np.log(1+loss):>7f} | CEloss_action: {CEL_size:>7f}, MSEloss: {RMSE_time} | batch: {batch} | sample: [{current:>5d}/{size:>5d}] | lr: {optimizer.param_groups[0]['lr']}"
            )
    (
        loss_rollingmean,
        lossRMSE_rollingmean,
        lossCEL_rollingmean,
    ) = (
        loss_rollingmean.detach().cpu().numpy().item(),
        lossRMSE_rollingmean.detach().cpu().numpy().item(),
        lossCEL_rollingmean.detach().cpu().numpy().item(),
    )

    print("epoch ended")
    print(
        f"Epoch loss:    mean: {loss_rollingmean:>7f}, ln(1+loss) mean: {np.log(1+loss_rollingmean):>7f}"
    )
    print(
        f"Epoch MSEloss: mean: {lossRMSE_rollingmean:>7f}, ln(1+loss) mean: {np.log(1+lossRMSE_rollingmean):>7f}"
    )
    print(
        f"Epoch CEloss_action:  mean: {lossCEL_rollingmean:>7f}, ln(1+loss) mean: {np.log(1+lossCEL_rollingmean):>7f}"
    )
    return (
        loss_rollingmean,
        lossRMSE_rollingmean,
        lossCEL_rollingmean,
    )


def model_train(
    epochs,
    train_loader,
    time_weight,
):
    torch.cuda.empty_cache()
    import gc

    gc.collect()

    encoder = EncoderNTPP().to(device)
    forecaster = ForecasterNTPP().to(device)

    # encoder.apply(utils.init_weights)
    # forecaster.apply(utils.init_weights)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(forecaster.parameters()),
        lr=0.001,
        eps=1e-16,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=3
    )

    train_loss_hist = pd.DataFrame(
        columns=[
            "epoch",
            "trn_L",
            "trn_MSEL",
            "trn_size",
        ]
    )
    time_start = datetime.now()

    early_stop_patience = 150
    patience_counter = 0
    best_train_loss = float("inf")
    best_encoding_state_dict = None
    epoch_train_loss = []
    epoch_train_acc = []

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        import gc

        gc.collect()
        print(f"Epoch {epoch}\n-------------------------------")
        train_loss = model_epoch(
            train_loader,
            train_loader,
            encoder,
            forecaster,
            optimizer,
            scheduler,
            "train",
            time_weight,
        )

        epochloss = pd.DataFrame(
            np.concatenate((np.array([epoch]), np.asarray(train_loss)))
        ).T
        # print("-----epochloss------")
        # print(epochloss)
        epochloss.columns = train_loss_hist.columns
        train_loss_hist = pd.concat([train_loss_hist, epochloss], ignore_index=True)

        # scheduler.step(valid_loss_value)
        # scheduler.step()

        print("train_loss : ", train_loss[0])
        print("best_train_loss : ", best_train_loss)
        if train_loss[0] < best_train_loss:
            best_train_loss = train_loss[0]
            best_epoch = epoch + 1
            best_encoding_state_dict = encoder.state_dict()  # 현재 상태 저장
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement: {patience_counter}/{early_stop_patience}")
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if optimizer.param_groups[0]["lr"] < 1e-7:
            break

    time_end = datetime.now()
    trainable_params_num = sum(
        p.numel() for p in encoder.parameters() if p.requires_grad
    ) + sum(p.numel() for p in forecaster.parameters() if p.requires_grad)
    train_time = time_end - time_start
    train_time = train_time.total_seconds()

    return (
        train_loss_hist,
        train_time,
        trainable_params_num,
        best_encoding_state_dict,
        encoder,
    )


if __name__ == "__main__":

    utils.seed_everything(0)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    (
        Alpha_train_loss,
        Alpha_train_time,
        Alpha_trainable_params_num,
        Alpha_model_parameter,
        Alpha_encoder,
    ) = model_train(
        epochs,
        Alpha_train_loader,
        Alpha_time_weight,
    )

    (
        HTTP_train_loss,
        HTTP_train_time,
        HTTP_trainable_params_num,
        HTTP_model_parameter,
        HTTP_encoder,
    ) = model_train(epochs, HTTP_train_loader, HTTP_time_weight)

    (
        Multi_train_loss,
        Multi_train_time,
        Multi_trainable_params_num,
        Multi_model_parameter,
        Multi_encoder,
    ) = model_train(
        epochs,
        Multi_train_loader,
        Multi_time_weight,
    )

    (
        Unlabeled_train_loss,
        Unlabeled_train_time,
        Unlabeled_trainable_params_num,
        Unlabeled_model_parameter,
        Unlabeled_encoder,
    ) = model_train(
        epochs,
        Unlabeled_train_loader,
        Unlabeled_time_weight,
    )

    out_dir = f"./results/first_stage/{WEEK}/{CONDITION}"

    def save_results(train_loss, model_params, train_time, num_params, label):
        os.makedirs(out_dir, exist_ok=True)
        train_time_num_params = pd.DataFrame([train_time, num_params]).T

        train_loss.to_csv(
            f"{out_dir}/{label}_train_loss.csv",
            index=False,
        )
        torch.save(
            model_params,
            f"{out_dir}/{label}_model.pt",
        )
        train_time_num_params.to_csv(
            f"{out_dir}/{label}_train_time_num_params.csv",
            index=False,
        )

    save_results(
        Alpha_train_loss,
        Alpha_model_parameter,
        Alpha_train_time,
        Alpha_trainable_params_num,
        "Alpha",
    )

    save_results(
        HTTP_train_loss,
        HTTP_model_parameter,
        HTTP_train_time,
        HTTP_trainable_params_num,
        "HTTP",
    )

    save_results(
        Multi_train_loss,
        Multi_model_parameter,
        Multi_train_time,
        Multi_trainable_params_num,
        "Multi",
    )

    save_results(
        Unlabeled_train_loss,
        Unlabeled_model_parameter,
        Unlabeled_train_time,
        Unlabeled_trainable_params_num,
        "Unlabeled",
    )
