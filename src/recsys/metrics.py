from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

def recall_at_k(ranked_items: List[str], true_item: str, k: int) -> float:
    return 1.0 if true_item in ranked_items[:k] else 0.0

def ndcg_at_k(ranked_items: List[str], true_item: str, k: int) -> float:
    # single relevant item => DCG is 1/log2(rank+1) if hit else 0; IDCG is 1
    topk = ranked_items[:k]
    if true_item not in topk:
        return 0.0
    rank = topk.index(true_item) + 1
    return 1.0 / np.log2(rank + 1)

def aggregate_metrics(rows: List[Dict], k_list: List[int]) -> pd.DataFrame:
    out = []
    for k in k_list:
        rec = np.mean([r[f"recall@{k}"] for r in rows]) if rows else 0.0
        nd = np.mean([r[f"ndcg@{k}"] for r in rows]) if rows else 0.0
        out.append({"k": k, "recall": float(rec), "ndcg": float(nd)})
    return pd.DataFrame(out)
