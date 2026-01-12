import os
import yaml
import pandas as pd
from recsys.data import load_raw, filter_min_counts, temporal_split, build_user_item_docs

def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    raw = cfg["data"]["raw_csv"]
    outdir = cfg["data"]["processed_dir"]
    os.makedirs(outdir, exist_ok=True)

    df = load_raw(raw)
    df = filter_min_counts(df, cfg["data"]["min_user_reviews"], cfg["data"]["min_item_reviews"])
    train_df, test_df = temporal_split(df, cfg["data"]["test_days"])

    user_docs_train, item_docs_train = build_user_item_docs(
        train_df,
        cfg["data"]["max_user_doc_chars"],
        cfg["data"]["max_item_doc_chars"]
    )

    # Save
    train_df.to_parquet(f"{outdir}/train.parquet", index=False)
    test_df.to_parquet(f"{outdir}/test.parquet", index=False)
    user_docs_train.to_parquet(f"{outdir}/user_docs.parquet", index=False)
    item_docs_train.to_parquet(f"{outdir}/item_docs.parquet", index=False)

    print("Prepared:")
    print(f"train={len(train_df)} test={len(test_df)} users={len(user_docs_train)} items={len(item_docs_train)}")

if __name__ == "__main__":
    main()
