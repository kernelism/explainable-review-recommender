import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from recsys.metrics import recall_at_k, ndcg_at_k, aggregate_metrics

def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    pdir = cfg["data"]["processed_dir"]

    train_df = pd.read_parquet(f"{pdir}/train.parquet")
    test_df = pd.read_parquet(f"{pdir}/test.parquet")
    user_docs = pd.read_parquet(f"{pdir}/user_docs.parquet")
    item_docs = pd.read_parquet(f"{pdir}/item_docs.parquet")

    # Only evaluate on users/items seen in train docs
    userset = set(user_docs["customer_id"].astype(str))
    itemset = set(item_docs["product_id"].astype(str))
    test_df = test_df[
        test_df["customer_id"].astype(str).isin(userset) &
        test_df["product_id"].astype(str).isin(itemset)
    ].copy()

    # Build doc maps
    ud = dict(zip(user_docs["customer_id"].astype(str), user_docs["user_doc"]))
    idoc = dict(zip(item_docs["product_id"].astype(str), item_docs["item_doc"]))

    # Candidate pool: sample items (for speed). You can increase later.
    all_items = item_docs["product_id"].astype(str).tolist()
    cand_n = min(cfg["eval"]["candidate_items_per_user"], len(all_items))
    k_list = cfg["eval"]["k_list"]

    model = SentenceTransformer("artifacts/models/two_tower", device="cuda")

    # Pre-encode ALL item docs once for evaluation over sampled candidates:
    # We'll still sample indices per user but reuse item embeddings.
    item_texts = [idoc[it] for it in all_items]
    item_emb = model.encode(item_texts, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

    rows = []
    rng = np.random.default_rng(0)

    # One test interaction per user (most recent in test)
    test_df = test_df.sort_values("review_date")
    last = test_df.groupby("customer_id").tail(1)

    for u, true_it in tqdm(list(zip(last["customer_id"].astype(str), last["product_id"].astype(str)))):
        utext = ud[u]
        uemb = model.encode([utext], batch_size=1, convert_to_numpy=True, normalize_embeddings=True)[0]

        cand_idx = rng.choice(len(all_items), size=cand_n, replace=False)
        # ensure true item included
        true_idx = all_items.index(true_it)
        if true_idx not in cand_idx:
            cand_idx[0] = true_idx

        sims = item_emb[cand_idx] @ uemb  # cosine since normalized -> inner product
        order = np.argsort(-sims)
        ranked = [all_items[cand_idx[i]] for i in order]

        row = {"user": u, "true_item": true_it}
        for k in k_list:
            row[f"recall@{k}"] = recall_at_k(ranked, true_it, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, true_it, k)
        rows.append(row)

    dfm = aggregate_metrics(rows, k_list)
    dfm.to_csv("artifacts/metrics.csv", index=False)
    print(dfm)

if __name__ == "__main__":
    main()
