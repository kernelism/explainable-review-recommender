from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers.datasets import NoDuplicatesDataLoader

@dataclass
class TwoTowerArtifacts:
    model_dir: str

def build_train_pairs(train_df, user_docs, item_docs) -> List[InputExample]:
    # Positive pairs: (user_doc, item_doc) for each (user,item) interaction
    ud = dict(zip(user_docs["customer_id"].astype(str), user_docs["user_doc"]))
    idoc = dict(zip(item_docs["product_id"].astype(str), item_docs["item_doc"]))

    pairs = []
    # Deduplicate interactions to reduce leakage and speed up training
    seen = set()
    for u, it in zip(train_df["customer_id"].astype(str), train_df["product_id"].astype(str)):
        key = (u, it)
        if key in seen:
            continue
        seen.add(key)
        utext = ud.get(u, "")
        itext = idoc.get(it, "")
        if utext and itext:
            pairs.append(InputExample(texts=[utext, itext]))
    return pairs

def train_two_tower(
    model_name: str,
    train_examples: List[InputExample],
    out_dir: str,
    batch_size: int,
    epochs: int,
    lr: float,
    warmup_ratio: float,
    mixed_precision: str = "fp16"
) -> TwoTowerArtifacts:
    model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    train_loss = losses.MultipleNegativesRankingLoss(model)  # positive pairs; negatives are other batch items [web:40] conceptually

    loader = NoDuplicatesDataLoader(train_examples, batch_size=batch_size)

    warmup_steps = int(len(loader) * epochs * warmup_ratio)

    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        use_amp=(mixed_precision.lower() in ["fp16", "bf16"]),
        show_progress_bar=True,
        output_path=out_dir
    )
    return TwoTowerArtifacts(model_dir=out_dir)
