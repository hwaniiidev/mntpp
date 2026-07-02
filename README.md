# NTPP ver1.1 — Experiment Execution Guide

## Project Structure

```
NTPP_ver1.1_proto_EJ/
├── train_first_size4.py        # Stage 1: NTPP encoder training
├── train_second_size4.py       # Stage 2: MLP classifier training
├── test_weight.py              # Compute time_weight ratios from Stage 1 results
├── config_size4.py             # Model hyperparameters (hidden_dim, attention, etc.)
├── config_label.yaml           # Label names, num_valid/num_test per class
├── models/
│   ├── NTPP_size4.py           # EncoderNTPP + ForecasterNTPP  (Stage 1)
│   ├── NTPP_size4_masking.py   # EncoderNTPP with padding mask  (Stage 2)
│   └── MLP.py                  # GroupedMLP1  (Stage 2 classifier)
├── utils/
│   ├── data_utils_bin4.py      # Data loading, pkt_size binning, train/valid/test split
│   └── utils.py                # Seed, plot, save utilities
├── results/
│   ├── first_stage/{WEEK}/{CONDITION}/   # Stage 1 output
│   └── second_stage/{WEEK}/{CONDITION}/ # Stage 2 output
└── weekly_log/                 # Weekly experiment notes
```

---

## Data Files

| File | Rows | `len` range | Note |
|------|------|-------------|------|
| `0901_30_K_5-10_T_1.0s.csv` | 7.88M | 5 – 10 | Original dataset |
| `merged_T1.0_kmax10.csv` | 1.80M | 10 only | Merged dataset, no padding |
| `merged_T0.5_kmax10.csv` | — | 10 only | T=0.5s variant |

---

## Full Experiment Flow

```
Stage 1  (train_first_size4.py, time_weight=1  →  condition: first_same_weights)
    ↓
test_weight.py  →  derive per-class time_weight ratios
    ↓
Stage 1  (train_first_size4.py, ratio-derived weights  →  condition: first_diff_weights)
    ↓
Stage 2  (train_second_size4.py, frozen Stage 1 encoders + MLP)
```

- **Stage 1** trains four independent NTPP encoders (one per class) via self-supervised next-packet prediction
- `first_same_weights` and `first_diff_weights` are two **conditions within Stage 1** — same script, different `--*_w` arguments
- **Stage 2** loads frozen Stage 1 encoders and trains an MLP classifier on top

---

## Default Parameters

### Window Size

| Parameter | Default |
|-----------|---------|
| `--window_size` | **4** |

`group_size = max_len − window_size = 9 − 4 = 5` (fixed number of window slots per connection in Stage 2).  
Must be identical in Stage 1 and Stage 2.

---

### Weight Ratio (Stage 1 time_weight)

Stage 1 loss function:
```
Loss = RMSE(pktTime) × time_weight  +  CrossEntropy(pktSize) × 1
```

| Parameter | `first_same_weights` | `first_diff_weights` |
|-----------|----------------------|----------------------|
| `--alpha_w` | **1** | **8** |
| `--http_w` | **1** | **5** |
| `--multi_w` | **1** | **7** |
| `--unlabeled_w` | **1** | **11** |

`first_diff_weights` values are derived by running `test_weight.py` on the `first_same_weights` results.

---

### MLP Architecture (Stage 2)

`GroupedMLP1` — 5-layer fully connected network.

```
Input  : 4 encoder outputs, each flattened over group_size slots
         → (group_size × feature_dim) × 4 = 5 × 5 × 4 = 100
Hidden : 100  (= feature_dim × output_size × group_size = 5 × 4 × 5)
Output : 4 classes
Activation : LeakyReLU (negative_slope=0.01)
Layers : fc1(100→100) → fc2 → fc3 → fc4 → fc5(100→4)
```

`feature_dim = 5` = encoder output dim (`pkt_time_linear(1) + pkt_size_embedding(4)`).

---

### Scheduler (Stage 2)

| Setting | Default |
|---------|---------|
| Scheduler | **None** |

Available options (defined but commented out in `train_second_size4.py`):

| Key | Scheduler |
|-----|-----------|
| `sched_none` ← **default** | No scheduler |
| `sched_plateau` | ReduceLROnPlateau |
| `sched_cosine` | CosineAnnealingLR |
| `sched_warm_restart` | CosineAnnealingWarmRestarts |
| `sched_multistep` | MultiStepLR |
| `sched_exp` | ExponentialLR |
| `sched_onecycle` | OneCycleLR |

To switch schedulers, uncomment the corresponding block in `train_second_size4.py` and reflect the choice in `--condition` (e.g. `..._sched_plateau`).

---

## Stage 1: NTPP Encoder Training

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--week` | `week3_may2026` | Output week folder |
| `--condition` | `w9_ws4` | Experiment condition name |
| `--window_size` | `4` | Sliding window size |
| `--alpha_w` | `9` | Alpha time_weight |
| `--http_w` | `5` | HTTP time_weight |
| `--multi_w` | `7` | Multi time_weight |
| `--unlabeled_w` | `11` | Unlabeled time_weight |

### Run Command

```bash
# first_same_weights: all time_weight = 1
python train_first_size4.py \
  --week <WEEK> \
  --condition first_same_weights \
  --window_size 4 \
  --alpha_w 1 --http_w 1 --multi_w 1 --unlabeled_w 1

# first_diff_weights: ratio-derived weights from test_weight.py
python train_first_size4.py \
  --week <WEEK> \
  --condition first_diff_weights \
  --window_size 4 \
  --alpha_w <A> --http_w <H> --multi_w <M> --unlabeled_w <U>
```

### Output

```
results/first_stage/{WEEK}/{CONDITION}/
├── Alpha_model.pt  /  Alpha_train_loss.csv  /  Alpha_train_time_num_params.csv
├── HTTP_model.pt   /  HTTP_train_loss.csv   /  ...
├── Multi_model.pt  /  Multi_train_loss.csv  /  ...
└── Unlabeled_model.pt  /  Unlabeled_train_loss.csv  /  ...
```

`train_loss.csv` columns: `epoch`, `trn_L`, `trn_MSEL`, `trn_size`

---

## test_weight.py: Derive Time Weight Ratios

Reads Stage 1 loss history (`first_same_weights`) and computes the ratio that balances time and size losses:

```
time_weight = min(size_loss) / min(MSEL_loss)
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--results_dir` | `./results/first_stage/week1_jun2026/first_same_weights` | Path to Stage 1 output directory |

### Run Command

```bash
python test_weight.py \
  --results_dir results/first_stage/<WEEK>/first_same_weights
```

### Output (stdout)

```
[Alpha]      min MSEL: 0.0443  min size: 0.3593  ratio: 8.10  →  suggested time_weight = 8
[HTTP]       min MSEL: 0.0174  min size: 0.0848  ratio: 4.87  →  suggested time_weight = 5
[Multi]      min MSEL: 0.0616  min size: 0.4309  ratio: 6.99  →  suggested time_weight = 7
[Unlabeled]  min MSEL: 0.0653  min size: 0.7149  ratio: 10.94 →  suggested time_weight = 11

Suggested --alpha_w/http_w/multi_w/unlabeled_w for first_diff_weights:
  --alpha_w 8 --http_w 5 --multi_w 7 --unlabeled_w 11
```

---

## Stage 2: MLP Classifier Training

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--week` | `week3_may2026` | Output week folder |
| `--stage1_week` | `None` → uses `--week` | Stage 1 weights week folder (specify only if different from `--week`) |
| `--stage1_condition` | `time_w_balanced` | Stage 1 condition name to load encoders from |
| `--condition` | `w9_ws4_mlp_A_sched_none` | Stage 2 experiment condition name |
| `--window_size` | `4` | Sliding window size (must match Stage 1) |

Stage 1 encoder path resolved as:
```
results/first_stage/{stage1_week}/{stage1_condition}/
```

### Run Command

```bash
python train_second_size4.py \
  --week <WEEK> \
  [--stage1_week <STAGE1_WEEK>] \
  --stage1_condition <STAGE1_CONDITION> \
  --condition <CONDITION> \
  --window_size 4
```

### Output

```
results/second_stage/{WEEK}/{CONDITION}/
├── best_mlp.pt
├── classification_report.csv
├── train_confusion_matrix.png  /  valid_confusion_matrix.png  /  test_confusion_matrix.png
├── Train_loss_plot.png  /  Train_acc_plot.png
└── Validation_loss_plot.png  /  Validation_acc_plot.png
```

### classification_report.csv Key Columns

| Column | Description |
|--------|-------------|
| `Best_epoch` | Epoch with best valid accuracy |
| `Test_Macro_F1` | Primary evaluation metric |
| `Test_Acc` | Overall test accuracy |
| `Test_Class{0–3}_F1` | Per-class F1 (0=Alpha, 1=HTTP, 2=Multi, 3=Unlabeled) |

---

## End-to-End Example

```bash
# Stage 1 — equal time weights
python train_first_size4.py \
  --week week4_jun2026 \
  --condition first_same_weights \
  --csv_path 2020.9-11_all_combined_T1.0_kmax10.csv \
  --alpha_w 1 --http_w 1 --multi_w 1 --unlabeled_w 1

# test_weight — derive calibrated weights from first_same_weights
python test_weight.py \
  --results_dir results/first_stage/week4_jun2026/first_same_weights
# → note the suggested --alpha_w / --http_w / --multi_w / --unlabeled_w

# Stage 1 — calibrated time weights  (can run in parallel with Stage 2 below)
python train_first_size4.py \
  --week week4_jun2026 \
  --condition first_diff_weights \
  --csv_path 2020.9-11_all_combined_T1.0_kmax10.csv \
  --alpha_w <A> --http_w <H> --multi_w <M> --unlabeled_w <U>

# Stage 2 — MLP on first_diff_weights encoders  (start after first_diff_weights is done)
python train_second_size4.py \
  --week week4_jun2026 \
  --stage1_condition first_diff_weights \
  --condition first_diff_weights_mlp_sched_none \
  --csv_path 2020.9-11_all_combined_T1.0_kmax10.csv 
```

> `first_diff_weights` Stage 1 and the `first_same_weights` Stage 2 can run in parallel.

---

## Other Fixed Settings

| Item | Value |
|------|-------|
| `max_len` | 9 |
| `n_streams` | 10,000 per class (cap) |
| `num_valid` / `num_test` | 200 / 100 per class |
| Stage 1 batch size | 10 |
| Stage 2 batch size | `group_size × 10 = 50` |
| Optimizer (both stages) | Adam, lr=0.001, eps=1e-16 |
| Stage 2 weight decay | 1e-4 |
| Early stopping patience | 150 epochs |
| Early stopping criterion | Stage 1: train loss / Stage 2: valid accuracy |
