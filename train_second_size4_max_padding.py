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

from config_size4 import *
import utils.data_utils_bin4 as data_utils
import utils.utils as utils
from models.NTPP_size4 import EncoderNTPP
from models.MLP import GroupedMLP1

from sklearn.metrics import classification_report

max_len = 20
epochs = 1000
window_size = 4
batch_size, num_workers = 10, 0  # batch 조정

max_subseq_num = max_len - window_size

csv_path = "20200901_30_k_min5_max20_T_2.0s_renamed_columns.csv"
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
        valid_len = int(df["len"].iloc[i])
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

# output_csv_path = "./train_sequences.csv"
# train_sequences.to_csv(output_csv_path, index=False)


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
        self.seq_ids = (
            train_sequences.iloc[self.valid_slice_idxn]["seqId"].unique().tolist()
        )

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, i):
        seq_id = self.seq_ids[i]
        # get row indices (as python list) where this sequence has valid slices
        seq_rows = train_sequences.loc[
            (train_sequences["seqId"] == seq_id) & (train_sequences["valid_slice_flag"])
        ].index.to_list()

        subsequences = []
        for j in seq_rows:
            # encode_train[j:j+window_size] 는 tensor slice (window_size, feature_dim)
            subseq = encode_train[j : j + window_size]
            subsequences.append(subseq.clone())  # clone to be safe

        # label as numpy -> torch.long. keep vector if multi-target, else scalar
        label_arr = train_sequences.loc[seq_rows[0], target_vars].to_numpy(dtype=int)
        label = torch.tensor(label_arr, dtype=torch.long)
        if label.numel() == 1:
            label = label.squeeze()  # scalar tensor

        return subsequences, label, int(seq_id)


class valid_data:
    def __init__(self, idx=idx_all_valid):
        self.idx = idx
        self.valid_slice_idxn = np.where(
            np.logical_and(self.idx, valid_sequences["valid_slice_flag"])
        )[0]
        self.seq_ids = (
            valid_sequences.iloc[self.valid_slice_idxn]["seqId"].unique().tolist()
        )

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, i):
        seq_id = self.seq_ids[i]
        # get row indices (as python list) where this sequence has valid slices
        seq_rows = valid_sequences.loc[
            (valid_sequences["seqId"] == seq_id) & (valid_sequences["valid_slice_flag"])
        ].index.to_list()

        subsequences = []
        for j in seq_rows:
            # encode_valid[j:j+window_size] 는 tensor slice (window_size, feature_dim)
            subseq = encode_valid[j : j + window_size]
            subsequences.append(subseq.clone())  # clone to be safe

        # label as numpy -> torch.long. keep vector if multi-target, else scalar
        label_arr = valid_sequences.loc[seq_rows[0], target_vars].to_numpy(dtype=int)
        label = torch.tensor(label_arr, dtype=torch.long)
        if label.numel() == 1:
            label = label.squeeze()  # scalar tensor

        return subsequences, label, int(seq_id)


class test_data:
    def __init__(self, idx=idx_all_test):
        self.idx = idx
        self.valid_slice_idxn = np.where(
            np.logical_and(self.idx, test_sequences["valid_slice_flag"])
        )[0]
        self.seq_ids = (
            test_sequences.iloc[self.valid_slice_idxn]["seqId"].unique().tolist()
        )

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, i):
        seq_id = self.seq_ids[i]
        # get row indices (as python list) where this sequence has valid slices
        seq_rows = test_sequences.loc[
            (test_sequences["seqId"] == seq_id) & (test_sequences["valid_slice_flag"])
        ].index.to_list()

        subsequences = []
        for j in seq_rows:
            # encode_test[j:j+window_size] 는 tensor slice (window_size, feature_dim)
            subseq = encode_test[j : j + window_size]
            subsequences.append(subseq.clone())  # clone to be safe

        # label as numpy -> torch.long. keep vector if multi-target, else scalar
        label_arr = test_sequences.loc[seq_rows[0], target_vars].to_numpy(dtype=int)
        label = torch.tensor(label_arr, dtype=torch.long)
        if label.numel() == 1:
            label = label.squeeze()  # scalar tensor

        return subsequences, label, int(seq_id)


train_dataset = train_data()
valid_dataset = valid_data()
test_dataset = test_data()

# print("train_dataset : ", len(train_dataset))
# print("valid_dataset : ", len(valid_dataset))
# print("test_dataset : ", len(test_dataset))
# train_dataset : 128150
# valid_dataset : 4000
# test_dataset : 2000


def collate_fn(batch):
    """
    batch: list of tuples (subsequences, label, seq_id)
           subsequences: list of tensors, each (window_size, feature_dim)
    Returns:
      flat_windows: Tensor (total_subseq, window_size, feature_dim)
      lengths: LongTensor (batch,)
      labels: LongTensor (batch,) or (batch, C)
      seq_ids: list of ints (len=batch)
    """
    lengths = []
    labels_list = []
    seq_ids = []
    flat_windows = []

    for subseqs, label, seq_id in batch:
        lengths.append(len(subseqs))
        labels_list.append(label)
        seq_ids.append(int(seq_id))
        for s in subseqs:
            flat_windows.append(s)

    # stack all windows (no padding)
    if len(flat_windows) == 0:
        # safety: if batch somehow empty
        flat_windows = torch.zeros(0, window_size, encode_train.shape[1])
    else:
        flat_windows = torch.stack(flat_windows, dim=0)  # (total_subseq, w, D)

    # build labels tensor (handle scalar labels vs vector labels)
    # if all labels are scalars -> produce shape (batch,), else (batch, C)
    if all(isinstance(l, torch.Tensor) and l.numel() == 1 for l in labels_list):
        labels = torch.tensor([int(l.item()) for l in labels_list], dtype=torch.long)
    else:
        # stack vectors; ensure each label is a tensor
        labels = torch.stack(
            [
                l if isinstance(l, torch.Tensor) else torch.tensor(l, dtype=torch.long)
                for l in labels_list
            ],
            dim=0,
        ).long()

    lengths = torch.tensor(lengths, dtype=torch.long)

    return flat_windows, labels, lengths, seq_ids


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


def pad_to_global_max(encoded, lengths, pad_value=0.0):
    """
    encoded: (total_subseq_in_batch, hidden_dim)
    lengths: (batch,)
    """
    batch_size = len(lengths)
    max_len = max_subseq_num
    hidden_dim = encoded.size(1)

    padded = torch.full(
        (batch_size, max_len, hidden_dim),
        pad_value,
        dtype=encoded.dtype,
        device=encoded.device,
    )

    idx = 0
    for i, l in enumerate(lengths):
        padded[i, :l] = encoded[idx : idx + l]
        idx += l

    return padded


torch.set_printoptions(threshold=float("inf"))


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

    for inputs, labels, lengths, _ in train_loader:
        optimizer.zero_grad()
        inputs, labels, lengths = (
            inputs.to(device),
            labels.to(device),
            lengths.to(device),
        )

        # print("inputs : ", inputs)
        # print("inputs.shape : ", inputs.shape)
        # print("labels : ", labels)
        # print("labels.shape : ", labels.shape)
        # print("lengths : ", lengths)
        # print("lengths : ", lengths.shape)

        Alpha_pred = Alpha_model(inputs)
        HTTP_pred = HTTP_model(inputs)
        Multi_pred = Multi_model(inputs)
        Normal_pred = Normal_model(inputs)

        Alpha_pred = pad_to_global_max(Alpha_pred, lengths)
        HTTP_pred = pad_to_global_max(HTTP_pred, lengths)
        Multi_pred = pad_to_global_max(Multi_pred, lengths)
        Normal_pred = pad_to_global_max(Normal_pred, lengths)

        # print("Alpha_pred : ", Alpha_pred)
        # print("Alpha_pred.shape : ", Alpha_pred.shape)

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
        for inputs, labels, lengths, _ in data_loader:
            inputs, labels, lengths = (
                inputs.to(device),
                labels.to(device),
                lengths.to(device),
            )

            Alpha_pred = Alpha_model(inputs)
            HTTP_pred = HTTP_model(inputs)
            Multi_pred = Multi_model(inputs)
            Normal_pred = Normal_model(inputs)

            Alpha_pred = pad_to_global_max(Alpha_pred, lengths)
            HTTP_pred = pad_to_global_max(HTTP_pred, lengths)
            Multi_pred = pad_to_global_max(Multi_pred, lengths)
            Normal_pred = pad_to_global_max(Normal_pred, lengths)

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
        for inputs, labels, lengths, _ in data_loader:
            inputs, labels, lengths = (
                inputs.to(device),
                labels.to(device),
                lengths.to(device),
            )
            Alpha_pred = Alpha_model(inputs)
            HTTP_pred = HTTP_model(inputs)
            Multi_pred = Multi_model(inputs)
            Normal_pred = Normal_model(inputs)

            Alpha_pred = pad_to_global_max(Alpha_pred, lengths)
            HTTP_pred = pad_to_global_max(HTTP_pred, lengths)
            Multi_pred = pad_to_global_max(Multi_pred, lengths)
            Normal_pred = pad_to_global_max(Normal_pred, lengths)

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

    Alpha_model = EncoderNTPP().to(device)
    Alpha_model.load_state_dict(torch.load(f"./first_diff_weights/Alpha_model.pt"))

    HTTP_model = EncoderNTPP().to(device)
    HTTP_model.load_state_dict(torch.load(f"./first_diff_weights/HTTP_model.pt"))

    Multi_model = EncoderNTPP().to(device)
    Multi_model.load_state_dict(torch.load(f"./first_diff_weights/Multi_model.pt"))

    Normal_model = EncoderNTPP().to(device)
    Normal_model.load_state_dict(torch.load(f"./first_diff_weights/Unlabeled_model.pt"))

    Alpha_model.eval()
    HTTP_model.eval()
    Multi_model.eval()
    Normal_model.eval()

    group_size = max_subseq_num
    feature_dim = 5
    output_size = 4
    hidden_size = feature_dim * output_size * group_size  #### hidden size 변경

    mlp = GroupedMLP1(group_size, feature_dim, hidden_size, output_size).to(device)

    mlp.apply(utils.init_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        mlp.parameters(), lr=0.001, eps=1e-16, weight_decay=1e-4
    )  # eps=1e-16
    ####  weight_decay=1e-4 L2 정규화 추가 (overfitting 방지)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", patience=5, factor=0.1, verbose=True
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[50, 100], gamma=0.1
    # )

    best_valid_acc = 0
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
                f"./second_results_diverse_ratio/best_mlp.pt",
            )

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}]")
            print(
                f"\tTrain Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Train F1 Score: {train_f1:.4f}"
            )
            print(
                f"\tValid Loss: {valid_loss_:.5f} | Valid Acc: {valid_accuracy:5f} | Valid F1 Score: {valid_f1:.4f}"
            )

    utils.plot_loss(
        np.linspace(1, epochs, epochs).astype(int),
        epoch_train_loss,
        "Train",
        "second_results_diverse_ratio",
    )
    utils.plot_loss(
        np.linspace(1, epochs, epochs).astype(int),
        epoch_valid_loss,
        "Validation",
        "second_results_diverse_ratio",
    )
    utils.plot_acc(
        np.linspace(1, epochs, epochs).astype(int),
        epoch_train_acc,
        "Train",
        "second_results_diverse_ratio",
    )
    utils.plot_acc(
        np.linspace(1, epochs, epochs).astype(int),
        epoch_valid_acc,
        "Validation",
        "second_results_diverse_ratio",
    )

    mlp.load_state_dict(torch.load(f"./second_results_diverse_ratio/best_mlp.pt"))

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
        "second_results_diverse_ratio",
    )
    utils.plot_cm(
        y_valid_list,
        y_valid_pred_list,
        "valid",
        label_dict["label"],
        "second_results_diverse_ratio",
    )
    utils.plot_cm(
        y_test_list,
        y_test_pred_list,
        "test",
        label_dict["label"],
        "second_results_diverse_ratio",
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
        "second_results_diverse_ratio",
        best_result,
        train_report_dict,
        valid_report_dict,
        test_report_dict,
    )


if __name__ == "__main__":

    utils.seed_everything(0)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_and_evaluate_model(epochs, train_loader, valid_loader, test_loader)
