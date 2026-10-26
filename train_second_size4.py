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
_parser.add_argument("--valid_csv_path", type=str, default=None)  # 달별 split: valid 파일 (10월)
_parser.add_argument("--test_csv_path", type=str, default=None)   # 달별 split: test 파일 (11월)
_parser.add_argument("--n_streams", type=int, default=10000)      # train 클래스당 최대 개수
_parser.add_argument("--num_valid", type=int, default=None)       # None이면 config_label.yaml 값 사용
_parser.add_argument("--num_test", type=int, default=None)        # None이면 config_label.yaml 값 사용
_parser.add_argument("--max_len", type=int, default=9)
_parser.add_argument("--epochs", type=int, default=1000)
_parser.add_argument("--seed", type=int, default=0)
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
epochs = _args.epochs
window_size = _args.window_size
batch_size, num_workers = (max_len - window_size) * 10, 0  # batch 조정

csv_path = _args.csv_path
n_streams = _args.n_streams  # --n_streams 값 사용(안 넣으면 10000), 부족한 클래스는 있는 만큼 최대한 사용

label_dict = yaml.safe_load(open("config_label.yaml", "r"))
# --num_valid 1000 --num_test 1000을 넣으면 valid/test 개수를 바꿈 (안 넣으면  기존 값 200/100 사용)
if _args.num_valid is not None:
    label_dict["num_valid"] = _args.num_valid
if _args.num_test is not None:
    label_dict["num_test"] = _args.num_test

valid_pkt_time_columns = [f"pktTime{i}" for i in range(max_len)]
valid_pkt_size_columns = [f"pktSize{i}" for i in range(max_len)]

# 터미널에서 받은 10월, 11월 파일 경로를 data_utils_bin4.py로 전달 (달별 split을 위해)
train, valid, test = data_utils.define_data(csv_path, max_len, n_streams, label_dict, _args.valid_csv_path, _args.test_csv_path)

#### train, valid, test가 동일한지 확인해보기

# print(len(train))
# print(len(valid))
# print(len(test))


def concat_pktTime_pktSize(df):
    # Vectorized: flattens the first min(len, max_len) packets of every row, row by row
    # (same output as the previous per-row loop)
    valid_len = np.minimum(df["len"].to_numpy().astype(int), max_len)
    packet_mask = np.arange(max_len)[None, :] < valid_len[:, None]

    result_df = pd.DataFrame(
        {
            "seqId": np.repeat(np.arange(len(df)), valid_len),
            "pktTime": df[valid_pkt_time_columns].to_numpy()[packet_mask],
            "pktSize": df[valid_pkt_size_columns].to_numpy()[packet_mask],
            "label": np.repeat(df["taxonomy label"].to_numpy(), valid_len),
        }
    )

    return result_df


train_sequences = concat_pktTime_pktSize(train)
valid_sequences = concat_pktTime_pktSize(valid)
test_sequences = concat_pktTime_pktSize(test)

target_vars = ["label"]


def valid_slice_flag(df, window_size):
    # Vectorized: marks the last window_size rows before every sequence end as invalid
    # (same output as the previous per-row loop)
    seq_end = np.flatnonzero(df["seqId"].to_numpy()[:-1] != df["seqId"].to_numpy()[1:])
    seq_end = np.append(seq_end, len(df) - 1)

    invalid_count = np.zeros(len(df) + 1, dtype=int)
    np.add.at(invalid_count, np.maximum(seq_end - window_size + 1, 0), 1)
    np.add.at(invalid_count, seq_end + 1, -1)
    df["valid_slice_flag"] = np.cumsum(invalid_count[:-1]) == 0
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

# label / seqId as arrays, so __getitem__ doesn't build a pandas row per sample
label_train = torch.from_numpy(train_sequences[target_vars].to_numpy(dtype="int"))
label_valid = torch.from_numpy(valid_sequences[target_vars].to_numpy(dtype="int"))
label_test = torch.from_numpy(test_sequences[target_vars].to_numpy(dtype="int"))
seq_id_train = train_sequences["seqId"].to_numpy()
seq_id_valid = valid_sequences["seqId"].to_numpy()
seq_id_test = test_sequences["seqId"].to_numpy()

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
        y = label_train[j + window_size]
        seq_id = seq_id_train[j + window_size]
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
        y = label_valid[j + window_size]
        seq_id = seq_id_valid[j + window_size]
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
        y = label_test[j + window_size]
        seq_id = seq_id_test[j + window_size]
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


def load_batches(data_loader):
    # Read a loader once and keep its batches on the device
    # (the loaders don't shuffle, so every epoch sees exactly these batches)
    batches = []
    for inputs, labels, lengths in data_loader:
        batches.append((inputs.to(device), labels.to(device).squeeze(1)))
    return batches


def run_encoders(
    batches, Alpha_model, HTTP_model, Multi_model, Normal_model, grad_enabled
):
    # The Stage 1 encoders are frozen (eval mode) and the batches are fixed,
    # so their outputs are the same every epoch: compute them once and reuse them.
    # grad_enabled mirrors the autograd mode of the previous per-epoch loop
    # (it can change which TransformerEncoderLayer kernel path runs), keeping outputs bit-identical
    encoded_batches = []
    with torch.set_grad_enabled(grad_enabled):
        for inputs, labels in batches:
            B, S, W, F = inputs.shape
            inputs_3d = inputs.view(B * S, W, F)
            window_lengths = torch.full((B * S,), W, dtype=torch.long, device=device)

            Alpha_pred = Alpha_model(inputs_3d, window_lengths).view(B, S, -1).detach()
            HTTP_pred = HTTP_model(inputs_3d, window_lengths).view(B, S, -1).detach()
            Multi_pred = Multi_model(inputs_3d, window_lengths).view(B, S, -1).detach()
            Normal_pred = Normal_model(inputs_3d, window_lengths).view(B, S, -1).detach()

            encoded_batches.append(
                ((Alpha_pred, HTTP_pred, Multi_pred, Normal_pred), labels)
            )
    return encoded_batches


def sum_losses(losses):
    # One device sync per epoch; same summation order / values as the previous per-batch loss.item()
    total_loss = 0.0
    for batch_loss in torch.stack(losses).tolist():
        total_loss += batch_loss
    return total_loss


def to_list(tensors):
    return list(torch.cat(tensors).cpu().numpy())


def train_model(
    mlp,
    encoded_batches,
    criterion,
    optimizer,
):
    mlp.train()

    losses = []
    all_preds, all_labels = [], []

    for (Alpha_pred, HTTP_pred, Multi_pred, Normal_pred), labels in encoded_batches:
        optimizer.zero_grad()

        predictions = mlp(
            Alpha_pred,
            HTTP_pred,
            Multi_pred,
            Normal_pred,
        )

        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        losses.append(loss.detach())

        _, predicted_classes = torch.max(predictions, 1)

        all_preds.append(predicted_classes)
        all_labels.append(labels)

    train_loss = sum_losses(losses) / len(encoded_batches)
    all_preds, all_labels = to_list(all_preds), to_list(all_labels)
    train_accuracy = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, average="weighted")

    return train_loss, train_accuracy, train_f1


def evaluate_model(
    mlp,
    encoded_batches,
    criterion,
):
    mlp.eval()

    losses = []
    all_preds, all_labels = [], []

    with torch.no_grad():
        for (Alpha_pred, HTTP_pred, Multi_pred, Normal_pred), labels in encoded_batches:
            predictions = mlp(
                Alpha_pred,
                HTTP_pred,
                Multi_pred,
                Normal_pred,
            )

            loss = criterion(predictions, labels)

            losses.append(loss)

            _, predicted_classes = torch.max(predictions, 1)

            all_preds.append(predicted_classes)
            all_labels.append(labels)

        loss = sum_losses(losses) / len(encoded_batches)
        all_preds, all_labels = to_list(all_preds), to_list(all_labels)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

    return loss, accuracy, f1


def predict_model(mlp, encoded_batches, return_proba=False):
    mlp.eval()

    all_preds, all_labels = [], []
    all_probs = []  # AUC용 클래스별 확률

    with torch.no_grad():
        for (Alpha_pred, HTTP_pred, Multi_pred, Normal_pred), labels in encoded_batches:
            predictions = mlp(
                Alpha_pred,
                HTTP_pred,
                Multi_pred,
                Normal_pred,
            )

            _, predicted_classes = torch.max(predictions, 1)
            all_probs.append(torch.softmax(predictions, dim=1))  # AUC용

            all_preds.append(predicted_classes)
            all_labels.append(labels)

    all_preds, all_labels = to_list(all_preds), to_list(all_labels)
    if return_proba:
        return all_labels, all_preds, torch.cat(all_probs).cpu().numpy()
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

    train_batches = load_batches(train_loader)
    valid_batches = load_batches(valid_loader)
    test_batches = load_batches(test_loader)
    encoder_models = (Alpha_model, HTTP_model, Multi_model, Normal_model)
    train_encoded = run_encoders(train_batches, *encoder_models, grad_enabled=True)  # as in train_model
    valid_encoded = run_encoders(valid_batches, *encoder_models, grad_enabled=False)  # as in evaluate_model

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
            train_encoded,
            criterion,
            optimizer,
        )
        valid_loss, valid_accuracy, valid_f1 = evaluate_model(
            mlp,
            valid_encoded,
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

    y_train_list, y_train_pred_list, y_train_prob = predict_model(
        mlp,
        run_encoders(train_batches, *encoder_models, grad_enabled=False),  # as in predict_model
        return_proba=True,  # AUC 계산용 확률도 받음
    )
    y_valid_list, y_valid_pred_list, y_valid_prob = predict_model(
        mlp,
        run_encoders(valid_batches, *encoder_models, grad_enabled=False),  # as in predict_model
        return_proba=True,  # AUC 계산용 확률도 받음
    )
    y_test_list, y_test_pred_list, y_test_prob = predict_model(
        mlp,
        run_encoders(test_batches, *encoder_models, grad_enabled=False),  # as in predict_model
        return_proba=True,  # AUC 계산용 확률도 받음
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
    # AUC (one-vs-rest, 클래스별 + macro)
    auc_dicts = {
        "Train": utils.compute_auc(y_train_list, y_train_prob),
        "Valid": utils.compute_auc(y_valid_list, y_valid_prob),
        "Test": utils.compute_auc(y_test_list, y_test_prob),
    }
    print("AUC (macro) - Train: {}, Valid: {}, Test: {}".format(
        auc_dicts["Train"]["macro"], auc_dicts["Valid"]["macro"], auc_dicts["Test"]["macro"]))

    utils.save_classification_report(
        OUT_DIR,
        best_result,
        train_report_dict,
        valid_report_dict,
        test_report_dict,
        auc_dicts,
    )


if __name__ == "__main__":

    utils.seed_everything(_args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_and_evaluate_model(epochs, train_loader, valid_loader, test_loader)



