import numpy as np
import pandas as pd

CONCEPTS = "data/test_concepts_manual.csv"
CAPTIONS = "data/test_captions.csv"
N = 2000
SEED = 42
MIN_PER_GROUP = 3
OUT = "data/subset_2000.csv"

rng = np.random.default_rng(SEED)

df = pd.read_csv(CONCEPTS).merge(pd.read_csv(CAPTIONS), on="ID")
df["group"] = df["CUIs"].fillna("(none)").apply(lambda c: ";".join(sorted(c.split(";"))))  
df["words"] = df["Caption"].astype(str).str.split().str.len()
sizes = df.groupby("group").size().to_dict()
print(sizes)
assert N >= sum(min(MIN_PER_GROUP, s) for s in sizes.values()), "N замалий для MIN_PER_GROUP"

take = {}
rest = dict(sizes)
print(rest)
while True:
    budget = N - sum(take.values())
    quota = {g: budget * s / sum(rest.values()) for g, s in rest.items()}
    small = [g for g in rest if quota[g] < min(MIN_PER_GROUP, rest[g])]
    if not small:
        break
    for g in small:  
        take[g] = min(MIN_PER_GROUP, rest.pop(g))
for g in quota:
    take[g] = int(quota[g])
leftover = N - sum(take.values())  
for g in sorted(quota, key=lambda g: quota[g] % 1, reverse=True)[:leftover]:
    take[g] += 1

parts = []
for g in sorted(take):
    rows = df[df.group == g]
    k = take[g]
    if k < len(rows):
        rows = rows.assign(tie=rng.random(len(rows))).sort_values(["words", "tie"])
        positions = np.floor((np.arange(k) + rng.random()) * len(rows) / k).astype(int)
        rows = rows.iloc[positions]
    rows = rows.sample(frac=1, random_state=int(rng.integers(2**31)))  
    parts.append(rows.assign(key=(np.arange(len(rows)) + rng.random()) / len(rows)))

subset = pd.concat(parts).sort_values("key")
assert len(subset) == N and subset["ID"].is_unique
subset[["ID", "CUIs"]].to_csv(OUT, index=False)


def concept_share(d):
    return d["CUIs"].dropna().str.split(";").explode().value_counts() / len(d) * 100


check = pd.DataFrame({"датасет": concept_share(df), "вибірка": concept_share(subset)}).fillna(0)
check["різниця"] = check["вибірка"] - check["датасет"]
print(f"Вибрано {len(subset)} з {len(df)}, груп: {subset.group.nunique()} з {df.group.nunique()}\n")
print("Частка зображень з концептом, %:")
print(check.sort_values("датасет", ascending=False).round(2).to_string())
print(f"\nНайбільше відхилення: {check['різниця'].abs().max():.2f} п.п.")

print("\nДовжина підпису, слів:")
print(pd.DataFrame({"датасет": df.words.describe(), "вибірка": subset.words.describe()}).round(1).to_string())