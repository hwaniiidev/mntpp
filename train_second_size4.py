import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
import torch.optim as optim
from datetime import datetime

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--week", type=str, default="week3_may2026")
_parser.add_argument("--stage1_week", type=str, default=None)  # None이면 --week 값 사용
_parser.add_argument("--stage1_condition", type=str, default="time_w_balanced")
_parser.add_argument("--condition", type=str, default="w9_ws4_mlp_A_sched_none")
_parser.add_argument("--window_size", type=int, default=4)
_parser.add_argument("--csv_path", type=str, default="all_combined_T1.0_kmax10.csv")
_parser.add_argument("--max_len", type=int, default=9)
_parser.add_argument("--scheduler", type=str, default="none",
                     choices=["none", "reduce", "cosine", "cosine_wr", "multistep", "exponential", "onecycle"])
_parser.add_argument("--mlp_type", type=str, default="grouped",
                     choices=["grouped", "res", "dense", "se"])
_args, _ = _parser.parse_known_args()

STAGE1_WEEK = _args.stage1_week if _args.stage1_week is not None else _args.week
STAGE1_CONDITION = _args.stage1_condition
STAGE1_WEIGHTS_DIR = f"./results/first_stage/{STAGE1_WEEK}/{STAGE1_CONDITION}"

WEEK = _args.week
CONDITION = _args.condition
OUT_DIR = f"./results/second_stage/{WEEK}/{CONDITION}"

from config_size4 import *
import utils.data_utils_bin4 as data_utils
import utils.utils as utils
from models.NTPP_size4_masking import EncoderNTPP
from models.MLP import GroupedMLP1, ResGroupedMLP1, DenseGroupedMLP, SEGroupedMLP

from sklearn.metrics import classification_report

max_len = _args.max_len
epochs = 1000
window_size = _args.window_size
batch_size, num_workers = (max_len - window_size) * 10, 0  # batch 조정

csv_path = _args.csv_path
n_streams = 10000

label_dict = yaml.safe_load(open("config_label.yaml", "r"))

valid_pkt_time_columns = [f"pktTime{i}" for i in range(max_len)]
valid_pkt_size_columns = [f"pktSize{i}" for i in range(max_len)]

train, valid, test = data_utils.define_data(csv_path, max_len, n_streams, label_dict)

#### train, valid, test가 동일한지 확인해보기

# print(len(train))
# print(len(valid))
# print(len(test))


def concat_pktTime_pktSize(df):
    concat_time = []
    concat_size = []
    seq_id = []
    label = []

    label_values = df["taxonomy label"].values

    for i in range(len(df)):
        valid_len = min(int(df["len"].iloc[i]), max_len)
        pkt_time_values = df[valid_pkt_time_columns].iloc[i, :valid_len].values
        pkt_size_values = df[valid_pkt_size_columns].iloc[i, :valid_len].values

        concat_time.extend(pkt_time_values)
        concat_size.extend(pkt_size_values)
        seq_id.extend([i] * valid_len)
        label.extend([label_values[i]] * valid_len)

    result_df = pd.DataFrame(
        {
            "seqId": seq_id,
            "pktTime": concat_time,
            "pktSize": concat_size,
            "label": label,
        }
    )

    return result_df


train_sequences = concat_pktTime_pktSize(train)
valid_sequences = concat_pktTime_pktSize(valid)
test_sequences = concat_pktTime_pktSize(test)

target_vars = ["label"]


def valid_slice_flag(df, window_size):
    df["valid_slice_flag"] = True
    for i in range(len(df) - 1):
        if df.seqId[i] != df.seqId[i + 1]:
            df.loc[i + 1 - window_size : i, ["valid_slice_flag"]] = False
    df.loc[len(df) - window_size : len(df), ["valid_slice_flag"]] = False
    return df


train_sequences = valid_slice_flag(train_sequences, window_size)
valid_sequences = valid_slice_flag(valid_sequences, window_size)
test_sequences = valid_slice_flag(test_sequences, window_size)

# output_csv_path = "./test_sequences.csv"
# test_sequences.to_csv(output_csv_path, index=False)


def encode(df):
    df_tensor_time = torch.from_numpy(df["pktTime"].values)
    df_tensor_time = df_tensor_time.view(len(df_tensor_time), 1)

    df_tensor_size = torch.from_numpy(df["pktSize"].values)
    df_tensor_size = df_tensor_size.view(len(df_tensor_size), 1)

    encode_df = torch.cat((df_tensor_time, df_tensor_size), 1)

    return encode_df


encode_train = encode(train_sequences)
encode_valid = encode(valid_sequences)
encode_test = encode(test_sequences)

# print(encode_test)
# print(len(encode_test)) # 4000

idx_all_train = np.repeat(True, len(train_sequences))
idx_all_valid = np.repeat(True, len(valid_sequences))
idx_all_test = np.repeat(True, len(test_sequences))

# print("idx_all_train : ", len(idx_all_train))  # 256300
# print("idx_all_valid : ", len(idx_all_valid))  # 8000
# print("idx_all_test : ", len(idx_all_test))  # 4000


class train_data:
    def __init__(self, idx=idx_all_train):
        self.idx = idx
        self.valid_slice_idxn = np.where(
            np.logical_and(self.idx, train_sequences["valid_slice_flag"])
        )[0]

    def __len__(self):
        return int(np.sum(train_sequences.loc[self.idx, "valid_slice_flag"]))

    def __getitem__(self, i):
        j = self.valid_slice_idxn[i]
        x = encode_train[j : j + window_size]
        row = train_sequences.iloc[j + window_size]
        y = row.loc[target_vars]
        seq_id = row["seqId"]
        y = torch.from_numpy(y.to_numpy(dtype="int"))
        return x, y, seq_id


class valid_data:
    def __init__(self, idx=idx_all_valid):
        self.idx = idx
        self.valid_slice_idxn = np.where(
            np.logical_and(self.idx, valid_sequences["valid_slice_flag"])
        )[0]

    def __len__(self):
        return int(np.sum(valid_sequences.loc[self.idx, "valid_slice_flag"]))

    def __getitem__(self, i):
        j = self.valid_slice_idxn[i]
        x = encode_valid[j : j + window_size]
        row = valid_sequences.iloc[j + window_size]
        y = row.loc[target_vars]
        seq_id = row["seqId"]
        y = torch.from_numpy(y.to_numpy(dtype="int"))
        return x, y, seq_id


class test_data:
    def __init__(self, idx=idx_all_test):
        self.idx = idx
        self.test_slice_idxn = np.where(
            np.logical_and(self.idx, test_sequences["valid_slice_flag"])
        )[0]

    def __len__(self):
        return int(np.sum(test_sequences.loc[self.idx, "valid_slice_flag"]))

    def __getitem__(self, i):
        j = self.test_slice_idxn[i]
        x = encode_test[j : j + window_size]
        row = test_sequences.iloc[j + window_size]
        y = row.loc[target_vars]
        seq_id = row["seqId"]
        y = torch.from_numpy(y.to_numpy(dtype="int"))
        return x, y, seq_id


train_dataset = train_data()
valid_dataset = valid_data()
test_dataset = test_data()

# print("train_dataset : ", len(train_dataset))
# print("valid_dataset : ", len(valid_dataset))
# print("test_dataset : ", len(test_dataset))
# train_dataset : 128150
# valid_dataset : 4000
# test_dataset : 2000


def collate_fn(batch, pad_value=0.0):
    from collections import defaultdict

    seq_dict = defaultdict(list)
    label_dict = {}

    for subseq, label, seq_id in batch:
        seq_dict[seq_id].append(subseq)
        label_dict[seq_id] = label

    sequences = list(seq_dict.values())
    labels = [label_dict[seq_id] for seq_id in seq_dict.keys()]

    sequences = [torch.stack(subseqs, dim=0) for subseqs in sequences]

    lengths = [seq.shape[0] for seq in sequences]
    group_size = max_len - window_size  # 항상 고정 크기로 패딩

    padded = []
    for seq in sequences:
        pad_size = group_size - seq.shape[0]
        if pad_size > 0:
            pad = torch.full(
                (pad_size, seq.shape[1], seq.shape[2]), pad_value, dtype=seq.dtype
            )
            seq = torch.cat([seq, pad], dim=0)
        padded.append(seq)

    padded = torch.stack(padded, dim=0)
    lengths = torch.tensor(lengths)

    return padded, torch.stack(labels), lengths


train_loader = DataLoader(
    train_dataset,
    shuffle=False,  #### shuffle
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=True,
    collate_fn=collate_fn,
)
valid_loader = DataLoader(
    valid_dataset,
    shuffle=False,  #### shuffle
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=True,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    test_dataset,
    shuffle=False,  #### shuffle
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=True,
    collate_fn=collate_fn,
)


# print("train_loader : ", len(train_loader))
# print("valid_loader : ", len(valid_loader))
# print("test_loader : ", len(test_loader))
# train_loader :  2563 (drop_last=True)
# valid_loader :  80 (drop_last=True / False)
# test_loader :  40 (drop_last=True / False)


def train_model(
    mlp,
    train_loader,
    Alpha_model,
    HTTP_model,
    Multi_model,
    Normal_model,
    criterion,
    optimizer,
):
    mlp.train()

    total_loss = 0.0
    all_preds, all_labels = [], []

    for inputs, labels, lengths in train_loader:
        optimizer.zero_grad()
        inputs, labels, lengths = (
            inputs.to(device),
            labels.to(device).squeeze(1),
            lengths.to(device),
        )

        # if first_print:
        #     print("inputs : ", inputs)
        #     print("labels : ", labels)
        #     first_print = False
        #     break

        B, S, W, F = inputs.shape
        inputs_3d = inputs.view(B * S, W, F)
        window_lengths = torch.full((B * S,), W, dtype=torch.long, device=device)

        Alpha_pred = Alpha_model(inputs_3d, window_lengths).view(B, S, -1)
        HTTP_pred = HTTP_model(inputs_3d, window_lengths).view(B, S, -1)
        Multi_pred = Multi_model(inputs_3d, window_lengths).view(B, S, -1)
        Normal_pred = Normal_model(inputs_3d, window_lengths).view(B, S, -1)

        predictions = mlp(
            Alpha_pred,
            HTTP_pred,
            Multi_pred,
            Normal_pred,
        )

        # print("predictions : ", predictions)
        # print("labels : ", labels)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # probs = F.softmax(predictions, dim=1)
        _, predicted_classes = torch.max(predictions, 1)

        # print("predicted_classes : ", predicted_classes)
        # print("labels : ", grouped_labels)

        all_preds.extend(predicted_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        # print("all_preds :", all_preds)
        # print("all_labels :", all_labels)

    train_loss = total_loss / len(train_loader)
    train_accuracy = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, average="weighted")

    return train_loss, train_accuracy, train_f1


def evaluate_model(
    mlp,
    data_loader,
    Alpha_model,
    HTTP_model,
    Multi_model,
    Normal_model,
    criterion,
):
    mlp.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels, lengths in data_loader:
            inputs, labels, lengths = (
                inputs.to(device),
                labels.to(device).squeeze(1),
                lengths.to(device),
            )

            B, S, W, F = inputs.shape
            inputs_3d = inputs.view(B * S, W, F)
            window_lengths = torch.full((B * S,), W, dtype=torch.long, device=device)

            Alpha_pred = Alpha_model(inputs_3d, window_lengths).view(B, S, -1)
            HTTP_pred = HTTP_model(inputs_3d, window_lengths).view(B, S, -1)
            Multi_pred = Multi_model(inputs_3d, window_lengths).view(B, S, -1)
            Normal_pred = Normal_model(inputs_3d, window_lengths).view(B, S, -1)

            predictions = mlp(
                Alpha_pred,
                HTTP_pred,
                Multi_pred,
                Normal_pred,
            )

            loss = criterion(predictions, labels)

            total_loss += loss.item()

            _, predicted_classes = torch.max(predictions, 1)

            all_preds.extend(predicted_classes.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

    return loss, accuracy, f1


def predict_model(mlp, data_loader, Alpha_model, HTTP_model, Multi_model, Normal_model):
    mlp.eval()

    all_preds, all_labels = [], []

    # first_print = True

    with torch.no_grad():
        for inputs, labels, lengths in data_loader:
            inputs, labels, lengths = (
                inputs.to(device),
                labels.to(device).squeeze(1),
                lengths.to(device),
            )
            B, S, W, F = inputs.shape
            inputs_3d = inputs.view(B * S, W, F)
            window_lengths = torch.full((B * S,), W, dtype=torch.long, device=device)

            Alpha_pred = Alpha_model(inputs_3d, window_lengths).view(B, S, -1)
            HTTP_pred = HTTP_model(inputs_3d, window_lengths).view(B, S, -1)
            Multi_pred = Multi_model(inputs_3d, window_lengths).view(B, S, -1)
            Normal_pred = Normal_model(inputs_3d, window_lengths).view(B, S, -1)

            predictions = mlp(
                Alpha_pred,
                HTTP_pred,
                Multi_pred,
                Normal_pred,
            )

            # probs = F.softmax(predictions, dim=1)
            _, predicted_classes = torch.max(predictions, 1)

            all_preds.extend(predicted_classes.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # y_pred_list = [a.flatten().tolist() for a in all_preds]
        # y_list = [a.flatten().tolist() for a in all_labels]

        # y_pred_list = list(itertools.chain(*y_pred_list))
        # y_list = list(itertools.chain(*y_list))

    return all_labels, all_preds


def train_and_evaluate_model(epochs, train_loader, valid_loader, test_loader):
    torch.cuda.empty_cache()
    import gc

    gc.collect()

    os.makedirs(OUT_DIR, exist_ok=True)

    Alpha_model = EncoderNTPP().to(device)
    Alpha_model.load_state_dict(torch.load(f"{STAGE1_WEIGHTS_DIR}/Alpha_model.pt", weights_only=True))

    HTTP_model = EncoderNTPP().to(device)
    HTTP_model.load_state_dict(torch.load(f"{STAGE1_WEIGHTS_DIR}/HTTP_model.pt", weights_only=True))

    Multi_model = EncoderNTPP().to(device)
    Multi_model.load_state_dict(torch.load(f"{STAGE1_WEIGHTS_DIR}/Multi_model.pt", weights_only=True))

    Normal_model = EncoderNTPP().to(device)
    Normal_model.load_state_dict(torch.load(f"{STAGE1_WEIGHTS_DIR}/Unlabeled_model.pt", weights_only=True))

    Alpha_model.eval()
    HTTP_model.eval()
    Multi_model.eval()
    Normal_model.eval()

    group_size = max_len - window_size  ####
    feature_dim = 5
    output_size = 4
    hidden_size = feature_dim * output_size * group_size  #### hidden size 변경

    _mlp_cls = {"grouped": GroupedMLP1, "res": ResGroupedMLP1, "dense": DenseGroupedMLP, "se": SEGroupedMLP}[_args.mlp_type]
    mlp = _mlp_cls(group_size, feature_dim, hidden_size, output_size).to(device)

    mlp.apply(utils.init_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        mlp.parameters(), lr=0.001, eps=1e-16, weight_decay=1e-4
    )  # eps=1e-16
    ####  weight_decay=1e-4 L2 정규화 추가 (overfitting 방지)

    _sched = _args.scheduler
    if _sched == "reduce":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.1
        )
    elif _sched == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    elif _sched == "cosine_wr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2
        )
    elif _sched == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50, 100], gamma=0.1
        )
    elif _sched == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif _sched == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.01, total_steps=epochs
        )
    else:
        scheduler = None

    early_stop_patience = 150
    patience_counter = 0
    best_valid_acc = 0
    best_epoch = 0
    train_loss_ = float("inf")
    valid_loss_ = float("inf")
    epoch_train_loss = []
    epoch_valid_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for epoch in range(epochs):
        train_loss, train_accuracy, train_f1 = train_model(
            mlp,
            train_loader,
            Alpha_model,
            HTTP_model,
            Multi_model,
            Normal_model,
            criterion,
            optimizer,
        )
        valid_loss, valid_accuracy, valid_f1 = evaluate_model(
            mlp,
            valid_loader,
            Alpha_model,
            HTTP_model,
            Multi_model,
            Normal_model,
            criterion,
        )

        epoch_train_loss.append(train_loss)
        epoch_valid_loss.append(valid_loss)

        epoch_train_acc.append(train_accuracy)
        epoch_valid_acc.append(valid_accuracy)

        if best_valid_acc < valid_accuracy:
            best_valid_acc = valid_accuracy
            best_epoch = epoch + 1
            train_loss_ = train_loss
            valid_loss_ = valid_loss
            print(f"Best Epoch [{best_epoch:02}]")
            print(f"\tTrain Loss: {train_loss_:.5f} | Train Acc: {train_accuracy:5f}")
            print(f"\tValid Loss: {valid_loss_:.5f} | Valid Acc: {valid_accuracy:5f}")
            torch.save(
                mlp.state_dict(),
                f"{OUT_DIR}/best_mlp.pt",
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(valid_loss)
            else:
                scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}]")
            print(
                f"\tTrain Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Train F1 Score: {train_f1:.4f}"
            )
            print(
                f"\tValid Loss: {valid_loss_:.5f} | Valid Acc: {valid_accuracy:5f} | Valid F1 Score: {valid_f1:.4f}"
            )

    actual_epochs = len(epoch_train_loss)
    utils.plot_loss(
        np.linspace(1, actual_epochs, actual_epochs).astype(int),
        epoch_train_loss,
        "Train",
        OUT_DIR,
    )
    utils.plot_loss(
        np.linspace(1, actual_epochs, actual_epochs).astype(int),
        epoch_valid_loss,
        "Validation",
        OUT_DIR,
    )
    utils.plot_acc(
        np.linspace(1, actual_epochs, actual_epochs).astype(int),
        epoch_train_acc,
        "Train",
        OUT_DIR,
    )
    utils.plot_acc(
        np.linspace(1, actual_epochs, actual_epochs).astype(int),
        epoch_valid_acc,
        "Validation",
        OUT_DIR,
    )

    mlp.load_state_dict(torch.load(f"{OUT_DIR}/best_mlp.pt", weights_only=True))

    y_train_list, y_train_pred_list = predict_model(
        mlp,
        train_loader,
        Alpha_model,
        HTTP_model,
        Multi_model,
        Normal_model,
    )
    y_valid_list, y_valid_pred_list = predict_model(
        mlp,
        valid_loader,
        Alpha_model,
        HTTP_model,
        Multi_model,
        Normal_model,
    )
    y_test_list, y_test_pred_list = predict_model(
        mlp,
        test_loader,
        Alpha_model,
        HTTP_model,
        Multi_model,
        Normal_model,
    )

    utils.plot_cm(
        y_train_list,
        y_train_pred_list,
        "train",
        label_dict["label"],
        OUT_DIR,
    )
    utils.plot_cm(
        y_valid_list,
        y_valid_pred_list,
        "valid",
        label_dict["label"],
        OUT_DIR,
    )
    utils.plot_cm(
        y_test_list,
        y_test_pred_list,
        "test",
        label_dict["label"],
        OUT_DIR,
    )

    target_names = ["class 0", "class 1", "class 2", "class 3"]

    train_report_dict = classification_report(
        y_train_list,
        y_train_pred_list,
        target_names=target_names,
        output_dict=True,
        digits=5,
    )
    valid_report_dict = classification_report(
        y_valid_list,
        y_valid_pred_list,
        target_names=target_names,
        output_dict=True,
        digits=5,
    )
    test_report_dict = classification_report(
        y_test_list,
        y_test_pred_list,
        target_names=target_names,
        output_dict=True,
        digits=5,
    )

    best_result = {
        "best_train_loss": train_loss_,
        "best_valid_loss": valid_loss_,
        "best_epoch": best_epoch,
    }
    utils.save_classification_report(
        OUT_DIR,
        best_result,
        train_report_dict,
        valid_report_dict,
        test_report_dict,
    )


if __name__ == "__main__":

    utils.seed_everything(0)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_and_evaluate_model(epochs, train_loader, valid_loader, test_loader)



