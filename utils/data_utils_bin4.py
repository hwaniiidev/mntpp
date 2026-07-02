from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import numpy as np
import pandas as pd


def shuffle_df(df):
    df = shuffle(df, random_state=0)
    df.reset_index(drop=True, inplace=True)
    return df


def process_data(df, attack_dict, label_dict):
    label_list = df["taxonomy label"].unique().tolist()

    target_group = label_dict["label"]

    for i in label_list:
        if i not in target_group:
            attack_dict[i] = 0

    df = (
        df.groupby("taxonomy label")
        .apply(
            lambda df: df.sample(attack_dict[df.name], replace=False, random_state=0)
        )
        .reset_index(drop=True)
    )
    return df


def count_num_attacks(df, n_streams):

    attack_dict = {}

    for idx, name in enumerate(df["taxonomy label"].value_counts().index.tolist()):
        count = df["taxonomy label"].value_counts()[idx]
        min_count = min(n_streams, count)
        attack_dict.update({name: min_count})

    return attack_dict


def categorize_pkt_size(df, k):
    pkt_size_columns = [f"pktSize{i}" for i in range(k)]

    # Reference-based bin boundaries (CAIDA / GaTech)
    # Bin 0: 0-100, Bin 1: 101-576, Bin 2: 577-1500, Bin 3: 1500+
    conditions = [
        df[pkt_size_columns].values > 1500,   # Bin 3: 1500+
        df[pkt_size_columns].values >= 577,   # Bin 2: 577-1500
        df[pkt_size_columns].values >= 101,   # Bin 1: 101-576
    ]
    # EJ percentile-based (data-dependent, deprecated)
    # conditions = [
    #     df[pkt_size_columns].values >= 1501,
    #     df[pkt_size_columns].values >= 198,
    #     df[pkt_size_columns].values >= 52,
    # ]
    choices = [3, 2, 1]
    df[pkt_size_columns] = np.select(conditions, choices, default=0)

    df = shuffle_df(df)

    return df


def process_train_valid_test(df, data, attack_dict, label_dict):
    label_list = df["taxonomy label"].unique().tolist()
    target_group = label_dict["label"]

    if data == "Test":
        num = label_dict["num_test"]

    elif data == "Valid":
        num = label_dict["num_valid"]

    # Number of rows, that we want to be sampled from each category
    attack_dict.update((k, num) for k in attack_dict)

    for i in label_list:
        if i not in target_group:
            attack_dict[i] = 0

    df = df.groupby("taxonomy label").apply(
        lambda df: df.sample(attack_dict[df.name], replace=False, random_state=0)
    )

    return df


def prepare_targets(df, label_dict):
    attack_categories = [label_dict["label"]]

    print("Alpha : ", len(df[df["taxonomy label"] == "Alpha"]))
    print("HTTP : ", len(df[df["taxonomy label"] == "HTTP"]))
    print("Multi : ", len(df[df["taxonomy label"] == "Multi"]))
    print("Unlabeled", len(df[df["taxonomy label"] == "Unlabeled"]))

    ordinal_encoder = OrdinalEncoder(categories=attack_categories)

    y_label = df["taxonomy label"].to_numpy()
    y_label = ordinal_encoder.fit_transform(y_label.reshape(-1, 1))

    # df["taxonomy label"] = y_label.astype(int)
    # print("df after prepare_tragets : ", df[df["taxonomy label"] == 3])

    return y_label.astype(int)


def load_data(data, n_streams, label_dict, k):
    # Read Training csv file
    df = pd.read_csv(data)

    # df = df[df["len"] >= 5].reset_index(drop=True)

    if "c2s" in df.columns:
        df.drop(["c2s"], axis=1, inplace=True)

    df = shuffle_df(df)

    attack_dict = count_num_attacks(df, n_streams)

    print("attack_dict")
    print(attack_dict)
    df = process_data(df, attack_dict, label_dict)

    df = categorize_pkt_size(df, k)
    print("size categorized : ", df)

    df_test = process_train_valid_test(df, "Test", attack_dict, label_dict)
    test_idx = pd.Index([x[1] for x in df_test.index])

    df = df.drop(test_idx).reset_index(drop=True)

    df_val = process_train_valid_test(df, "Valid", attack_dict, label_dict)
    val_idx = pd.Index([x[1] for x in df_val.index])

    df = df.drop(val_idx).reset_index(drop=True)

    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    for idx, name in enumerate(df["taxonomy label"].value_counts().index.tolist()):
        print("Train", name)
        print("Counts:", df["taxonomy label"].value_counts()[idx])

    for idx, name in enumerate(df_val["taxonomy label"].value_counts().index.tolist()):
        print("Valid", name)
        print("Counts:", df_val["taxonomy label"].value_counts()[idx])

    for idx, name in enumerate(df_test["taxonomy label"].value_counts().index.tolist()):
        print("Test", name)
        print("Counts:", df_test["taxonomy label"].value_counts()[idx])

    pd.set_option("display.max_columns", None)
    print("======mine_df_head======")
    print(df.head())

    # create Xtrain, ytrain, Xtest and ytest
    # X_train = df.iloc[:, :-1].to_numpy()
    # y_train = df.iloc[:, -1].to_numpy()

    # X_valid = df_val.iloc[:, :-1].to_numpy()
    # y_valid = df_val.iloc[:, -1].to_numpy()

    # X_test = df_test.iloc[:, :-1].to_numpy()
    # y_test = df_test.iloc[:, -1].to_numpy()

    return df, df_val, df_test


def define_data(df, k_packets, n_streams, label_dict):

    # load data, feature scaling
    train, valid, test = load_data(df, n_streams, label_dict, k_packets)
    # X_train, X_valid, X_test = feature_scaling(X_train, X_valid, X_test, k_packets)
    # X_train = change_outlier(X_train)
    # X_valid = change_outlier(X_valid)
    # X_test = change_outlier(X_test)

    train["taxonomy label"] = prepare_targets(train, label_dict)
    valid["taxonomy label"] = prepare_targets(valid, label_dict)
    test["taxonomy label"] = prepare_targets(test, label_dict)

    return train, valid, test
