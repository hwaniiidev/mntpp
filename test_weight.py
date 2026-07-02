import pandas as pd
import argparse
import math

_parser = argparse.ArgumentParser()
_parser.add_argument("--results_dir", type=str,
                     default="./results/first_stage/week1_jun2026/first_same_weights")
_args, _ = _parser.parse_known_args()

labels = ["Alpha", "HTTP", "Multi", "Unlabeled"]
file_paths = [f"{_args.results_dir}/{label}_train_loss.csv" for label in labels]

print(f"Reading from: {_args.results_dir}\n")
suggested = {}
for label, path in zip(labels, file_paths):
    df = pd.read_csv(path)
    min_msel = df["trn_MSEL"].min()
    min_size = df["trn_size"].min()
    ratio = min_size / min_msel if min_msel != 0 else None

    print(f"[{label}]")
    print(f"  min MSEL (time loss): {min_msel:.6f}")
    print(f"  min size (size loss): {min_size:.6f}")
    if ratio is not None:
        rounded = round(ratio)
        suggested[label] = rounded
        print(f"  ratio (size/MSEL)  : {ratio:.4f}  →  suggested time_weight = {rounded}")
    else:
        print("  min_msel == 0, cannot compute ratio")
    print()

print("=" * 50)
print("Suggested --alpha_w/http_w/multi_w/unlabeled_w for first_diff_weights:")
print(f"  --alpha_w {suggested.get('Alpha','?')} --http_w {suggested.get('HTTP','?')} "
      f"--multi_w {suggested.get('Multi','?')} --unlabeled_w {suggested.get('Unlabeled','?')}")


# 20200901_30_k_min10_max50_T_1.0s(0~9)_renamed_columns.csv
# trn_MSEL's training loss min: 0.8569970726966858
# trn_size' training loss min: 0.6283161640167236
# ratio = min_msel / min_size: 0.7331602219359057 # 1

# trn_MSEL's training loss min: 0.1227468773722648
# trn_size' training loss min: 0.1346797496080398
# ratio = min_msel / min_size: 1.0972152814900957 # 2

# trn_MSEL's training loss min: 1.781364679336548
# trn_size' training loss min: 0.5181869268417358
# ratio = min_msel / min_size: 0.2908932308205019

# trn_MSEL's training loss min: 4.186462879180908
# trn_size' training loss min: 0.6809840202331543
# ratio = min_msel / min_size: 0.16266333654113052


# monthly k10_w7_size4
# ratio = min_msel / min_size: 12.048154702910177 => 13
# ratio = min_msel / min_size: 4.24193879179604 => 5
# ratio = min_msel / min_size: 8.530280318944689 => 9
# ratio = min_msel / min_size: 11.97359658020058 => 12

# 3-month k20_w5_size(0,52,63,420,1501)
# ratio = min_msel / min_size: 11.949344173942247 => 12
# ratio = min_msel / min_size: 14.978774981424023 => 15
# ratio = min_msel / min_size: 7.540529668531422 => 8
# ratio = min_msel / min_size: 16.100950580279875 => 17

# 3-month k10_w5_size(0,52,70,557,1501)
# ratio = min_msel / min_size: 10.308156891935035 => 11
# ratio = min_msel / min_size: 10.261945796558859 => 11
# ratio = min_msel / min_size: 8.72851886137797 => 9
# ratio = min_msel / min_size: 15.505516110829367 => 16

# weekly k10_w5_size(0,52,63,420,1501)
# ratio = min_msel / min_size: 15.384998612553405 => 16
# ratio = min_msel / min_size: 13.245216969179587 => 14
# ratio = min_msel / min_size: 7.691238990285603 => 8
# ratio = min_msel / min_size: 15.840272793385255 => 16

# 3-month k10_w5_size(0,52,63,420,1501)
# ratio = min_msel / min_size: 10.599962965958063 => 11
# ratio = min_msel / min_size: 10.241050616773137 => 11
# ratio = min_msel / min_size: 8.180281151478232 => 9
# ratio = min_msel / min_size: 15.696442426706872 => 16
