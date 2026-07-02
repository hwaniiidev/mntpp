import pandas as pd

df = pd.read_csv("20200901_30_k_min5_max20_T_1.0s_renamed_columns.csv")

count_series = df["len"].value_counts().sort_index()
ratio_series = df["len"].value_counts(normalize=True).sort_index() * 100

result = pd.DataFrame({"count": count_series, "ratio (%)": ratio_series.round(2)})

print(result)
print("\ntotal", df["len"].count())
