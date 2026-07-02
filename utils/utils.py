import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from collections import defaultdict


def seed_everything(seed: int):
    """
    Function to seed everything for reproducibility in PyTorch.
    Args:
        seed (int): The seed value to be set.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for NumPy's random number generator
    np.random.seed(seed)

    # Set seed for PyTorch's random number generator (both CPU and CUDA)
    torch.manual_seed(seed)

    # If using CUDA, ensure reproducibility there as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Ensure deterministic behavior
    torch.use_deterministic_algorithms(True)

    # Set PyTorch threads to avoid multithreading-related randomness
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Ensure CuDNN uses deterministic algorithms (might affect speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional: Ensure CUDA operations are synchronous
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.zeros_(m.bias)


def plot_loss(epochs: int, loss, data: str, result_path: str):
    """
    Function that plot and save loss curve as a png file

    Args:
        epochs (int): the number times that the model worked through the entire dataset
        loss (list[float]): loss values
        data (str): dataset type e.g., Train, Validation
        result_path (str): file path for saving loss plot
    """

    plt.figure(figsize=(10, 10))
    label_name = f"{data} Loss"
    plt.plot(epochs, loss, label=label_name)
    # plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.title("Loss Curve")
    plt.legend()
    plt.savefig(f"./{result_path}/{data}_loss_plot.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_acc(epochs: int, acc, data: str, result_path: str):
    """
    Function that plot and save accuracy curve as a png file

    Args:
        epochs (int): the number times that the model worked through the entire dataset
        acc (list[float]): accuracy values
        data (str): dataset type e.g., Train, Validation
        result_path (str): file path for saving accuracy plot
    """

    plt.figure(figsize=(10, 10))
    label_name = f"{data} accuracy"
    plt.plot(epochs, acc, label=label_name)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    # plt.title("Accuracy Curve")
    plt.legend()
    plt.savefig(f"{result_path}/{data}_acc_plot.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_cm(y_list, y_pred_list, data, categories, result_path: str):
    """
    Function that calculate, plot and save confusion matrix as a png file

    Args:
        y_list (list[float]): list of true labels
        y_pred_list (list[float]): list of prediction labels
        data (str): dataset type e.g., Train, Validation
        categories (list[str]): list of labels
        result_path (str): file path for saving confusion matrix
    """
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.5)
    cm = confusion_matrix(y_list, y_pred_list)
    # categories = ["Unlabeled", "HTTP", "Alpha", "Multi"]
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = [
        "{0:.2%}".format(value)
        for value in (cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]).flatten()
    ]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(4, 4)  # (4, 4)
    heatmap = sns.heatmap(
        cm,
        annot=labels,
        cmap="Blues",
        fmt="",
        xticklabels=categories,
        yticklabels=categories,
    )
    # heatmap = sns.heatmap(cm, annot=labels,  fmt='',xticklabels=categories,yticklabels=categories)
    plt.xlabel("Predicted label", fontsize=20)
    plt.ylabel("True label", fontsize=20)
    plt.savefig(
        f"./{result_path}/{data}_confusion_matrix.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def save_classification_report(
    result_path,
    best_result,
    train_report_dict,
    valid_report_dict,
    test_report_dict,
):
    i = 0
    dic = defaultdict(lambda: defaultdict(dict))
    dic[i]["Best_epoch"] = best_result["best_epoch"]
    dic[i]["Best_Train_loss"] = best_result["best_train_loss"]
    dic[i]["Best_Valid_loss"] = best_result["best_valid_loss"]

    dic[i]["Train_Macro_F1"] = train_report_dict["macro avg"]["f1-score"]
    dic[i]["Train_Class0_F1"] = train_report_dict["class 0"]["f1-score"]
    dic[i]["Train_Class1_F1"] = train_report_dict["class 1"]["f1-score"]
    dic[i]["Train_Class2_F1"] = train_report_dict["class 2"]["f1-score"]
    dic[i]["Train_Class3_F1"] = train_report_dict["class 3"]["f1-score"]
    dic[i]["Train_Macro_precision"] = train_report_dict["macro avg"]["precision"]
    dic[i]["Train_Class0_precision"] = train_report_dict["class 0"]["precision"]
    dic[i]["Train_Class1_precision"] = train_report_dict["class 1"]["precision"]
    dic[i]["Train_Class2_precision"] = train_report_dict["class 2"]["precision"]
    dic[i]["Train_Class3_precision"] = train_report_dict["class 3"]["precision"]
    dic[i]["Train_Macro_recall"] = train_report_dict["macro avg"]["recall"]
    dic[i]["Train_Class0_recall"] = train_report_dict["class 0"]["recall"]
    dic[i]["Train_Class1_recall"] = train_report_dict["class 1"]["recall"]
    dic[i]["Train_Class2_recall"] = train_report_dict["class 2"]["recall"]
    dic[i]["Train_Class3_recall"] = train_report_dict["class 3"]["recall"]
    dic[i]["Train_Acc"] = train_report_dict["accuracy"]

    dic[i]["Valid_Macro_F1"] = valid_report_dict["macro avg"]["f1-score"]
    dic[i]["Valid_Class0_F1"] = valid_report_dict["class 0"]["f1-score"]
    dic[i]["Valid_Class1_F1"] = valid_report_dict["class 1"]["f1-score"]
    dic[i]["Valid_Class2_F1"] = valid_report_dict["class 2"]["f1-score"]
    dic[i]["Valid_Class3_F1"] = valid_report_dict["class 3"]["f1-score"]
    dic[i]["Valid_Macro_precision"] = valid_report_dict["macro avg"]["precision"]
    dic[i]["Valid_Class0_precision"] = valid_report_dict["class 0"]["precision"]
    dic[i]["Valid_Class1_precision"] = valid_report_dict["class 1"]["precision"]
    dic[i]["Valid_Class2_precision"] = valid_report_dict["class 2"]["precision"]
    dic[i]["Valid_Class3_precision"] = valid_report_dict["class 3"]["precision"]
    dic[i]["Valid_Macro_recall"] = valid_report_dict["macro avg"]["recall"]
    dic[i]["Valid_Class0_recall"] = valid_report_dict["class 0"]["recall"]
    dic[i]["Valid_Class1_recall"] = valid_report_dict["class 1"]["recall"]
    dic[i]["Valid_Class2_recall"] = valid_report_dict["class 2"]["recall"]
    dic[i]["Valid_Class3_recall"] = valid_report_dict["class 3"]["recall"]
    dic[i]["Valid_Acc"] = valid_report_dict["accuracy"]

    dic[i]["Test_Macro_F1"] = test_report_dict["macro avg"]["f1-score"]
    dic[i]["Test_Class0_F1"] = test_report_dict["class 0"]["f1-score"]
    dic[i]["Test_Class1_F1"] = test_report_dict["class 1"]["f1-score"]
    dic[i]["Test_Class2_F1"] = test_report_dict["class 2"]["f1-score"]
    dic[i]["Test_Class3_F1"] = test_report_dict["class 3"]["f1-score"]
    dic[i]["Test_Macro_precision"] = test_report_dict["macro avg"]["precision"]
    dic[i]["Test_Class0_precision"] = test_report_dict["class 0"]["precision"]
    dic[i]["Test_Class1_precision"] = test_report_dict["class 1"]["precision"]
    dic[i]["Test_Class2_precision"] = test_report_dict["class 2"]["precision"]
    dic[i]["Test_Class3_precision"] = test_report_dict["class 3"]["precision"]
    dic[i]["Test_Macro_recall"] = test_report_dict["macro avg"]["recall"]
    dic[i]["Test_Class0_recall"] = test_report_dict["class 0"]["recall"]
    dic[i]["Test_Class1_recall"] = test_report_dict["class 1"]["recall"]
    dic[i]["Test_Class2_recall"] = test_report_dict["class 2"]["recall"]
    dic[i]["Test_Class3_recall"] = test_report_dict["class 3"]["recall"]
    dic[i]["Test_Acc"] = test_report_dict["accuracy"]

    df = pd.DataFrame.from_dict(dic, orient="index")
    df.to_csv(f"./{result_path}/classification_report.csv", index=False)
