import os
import yaml
import pandas as pd
from recsys.model import build_train_pairs, train_two_tower

def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    pdir = cfg["data"]["processed_dir"]

    train_df = pd.read_parquet(f"{pdir}/train.parquet")
    user_docs = pd.read_parquet(f"{pdir}/user_docs.parquet")
    item_docs = pd.read_parquet(f"{pdir}/item_docs.parquet")

    examples = build_train_pairs(train_df, user_docs, item_docs)
    os.makedirs("artifacts/models", exist_ok=True)

    train_two_tower(
        model_name=cfg["train"]["model_name"],
        train_examples=examples,
        out_dir="artifacts/models/two_tower",
        batch_size=cfg["train"]["batch_size"],
        epochs=cfg["train"]["epochs"],
        lr=cfg["train"]["lr"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        mixed_precision=cfg["train"]["mixed_precision"],
    )

    print(f"Saved model to artifacts/models/two_tower with {len(examples)} pairs")

if __name__ == "__main__":
    main()
