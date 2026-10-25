import json
import re
from pathlib import Path

import evaluate
import pandas as pd
from bert_score import score as bert_score
from pycocoevalcap.cider.cider import Cider

PRED_FILE = "predictions/Hulu-Med-4B__caption_v1.jsonl"
REFS_CSV = "test_captions.csv"
ID_COL = "ID"
CAPTION_COL = "Caption"
OUT_DIR = "results"


def norm(s):
    return " ".join(re.sub(r"[^\w\s]", " ", str(s).lower()).split())


def load_pairs(jsonl, refs):
    preds = pd.read_json(jsonl, lines=True).drop_duplicates("image_id", keep="last")
    preds["image_id"] = preds["image_id"].astype(str)
    df = preds.merge(refs, on="image_id", how="left").rename(columns={"caption": "caption_pred"})
    missing = df.loc[df.caption_ref.isna(), "image_id"].tolist()
    if missing:
        raise ValueError(f"{jsonl}: {len(missing)} зображень без еталону, напр. {missing[:5]}")
    return df


def score(df):
    P = [norm(x) for x in df.caption_pred]
    R = [norm(x) for x in df.caption_ref]

    df["rougeL"] = evaluate.load("rouge").compute(
        predictions=P, references=R, use_aggregator=False)["rougeL"]

    _, per_cider = Cider().compute_score({i: [r] for i, r in enumerate(R)},
                                         {i: [p] for i, p in enumerate(P)})
    df["cider"] = per_cider

    _, _, f1 = bert_score(df.caption_pred.tolist(), df.caption_ref.tolist(), lang="en")
    df["bertscore_f1"] = f1.tolist()

    df["len_pred_words"] = df.caption_pred.str.split().str.len()
    df["len_ref_words"] = df.caption_ref.str.split().str.len()

    stats = {"model": df.model.iloc[0], "prompt_id": df.prompt_id.iloc[0], "n_samples": len(df)}

    bleu = evaluate.load("bleu")
    for n in range(1, 5):
        stats[f"bleu{n}"] = bleu.compute(
            predictions=P, references=[[r] for r in R], max_order=n)["bleu"]
    for c in ["rougeL", "cider", "bertscore_f1", "len_pred_words", "len_ref_words"]:
        stats[c] = df[c].mean()

    stats = {k: round(float(v), 4) if isinstance(v, float) else v for k, v in stats.items()}
    return df, stats


def main():
    refs = pd.read_csv(REFS_CSV).rename(columns={ID_COL: "image_id", CAPTION_COL: "caption_ref"})
    refs = refs[["image_id", "caption_ref"]]
    refs["image_id"] = refs["image_id"].astype(str)

    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    df = load_pairs(PRED_FILE, refs)
    df, stats = score(df)

    name = f"{stats['model']}__{stats['prompt_id']}"
    (out / f"metrics_{name}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    df.to_csv(out / f"per_sample_{name}.csv", index=False)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()