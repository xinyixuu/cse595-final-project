
import pandas as pd
import os

root = "./ethics"
subsets = ["commonsense", "utilitarianism", "virtue", "deontology", "justice"]

stats = []

for sub in subsets:
    train_path = os.path.join(root, sub, f"cm_train.csv")
    test_path = os.path.join(root, sub, f"cm_test.csv")

    for split, path in [("train", train_path), ("test", test_path)]:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        n = len(df)
        avg_len = df["input"].astype(str).apply(len).mean()
        label_ratio = df["label"].value_counts(normalize=True).to_dict()
        ratio_0 = label_ratio.get(0, 0)
        ratio_1 = label_ratio.get(1, 0)
        stats.append({
            "subset": sub,
            "split": split,
            "num_samples": n,
            "avg_length": round(avg_len, 1),
            "label_0_ratio": round(ratio_0, 3),
            "label_1_ratio": round(ratio_1, 3),
        })


df_stats = pd.DataFrame(stats)
print(df_stats.to_string(index=False))