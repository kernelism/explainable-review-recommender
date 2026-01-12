from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np

REQUIRED_COLS = [
    "customer_id", "product_id", "product_title", "product_category",
    "star_rating", "verified_purchase", "review_headline", "review_body", "review_date"
]

def _clean_text(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x

def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    df = df.copy()
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df = df.dropna(subset=["review_date", "customer_id", "product_id"])
    df["star_rating"] = pd.to_numeric(df["star_rating"], errors="coerce").fillna(0).astype(int)
    df["verified_purchase"] = df["verified_purchase"].astype(str).str.upper().isin(["Y", "TRUE", "1"])
    df["review_headline"] = df["review_headline"].map(_clean_text)
    df["review_body"] = df["review_body"].map(_clean_text)
    df["product_title"] = df["product_title"].map(_clean_text)
    df["product_category"] = df["product_category"].map(_clean_text)
    return df

def make_review_text(df: pd.DataFrame) -> pd.Series:
    # short structured prefix helps explanations and embedding consistency
    prefix = (
        "TITLE: " + df["product_title"].fillna("") +
        " | CATEGORY: " + df["product_category"].fillna("") +
        " | STARS: " + df["star_rating"].astype(str) +
        " | VERIFIED: " + df["verified_purchase"].astype(int).astype(str) +
        " | HEADLINE: " + df["review_headline"].fillna("")
    )
    body = df["review_body"].fillna("")
    return (prefix + " | BODY: " + body).str.strip()

def clip_chars(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars]

def build_user_item_docs(
    df: pd.DataFrame,
    max_user_doc_chars: int,
    max_item_doc_chars: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["review_text"] = make_review_text(df)

    # Sort so most recent comes last; we will append in chronological order
    df = df.sort_values("review_date")

    # User docs = concat all review_text written by user
    user_docs = (
        df.groupby("customer_id")["review_text"]
        .apply(lambda xs: " [SEP] ".join(xs.tolist()))
        .reset_index()
        .rename(columns={"review_text": "user_doc"})
    )
    user_docs["user_doc"] = user_docs["user_doc"].map(lambda x: clip_chars(x, max_user_doc_chars))

    # Item docs = concat all review_text about item
    item_docs = (
        df.groupby("product_id")["review_text"]
        .apply(lambda xs: " [SEP] ".join(xs.tolist()))
        .reset_index()
        .rename(columns={"review_text": "item_doc"})
    )
    item_docs["item_doc"] = item_docs["item_doc"].map(lambda x: clip_chars(x, max_item_doc_chars))

    return user_docs, item_docs

def filter_min_counts(df: pd.DataFrame, min_user_reviews: int, min_item_reviews: int) -> pd.DataFrame:
    df = df.copy()
    ucnt = df.groupby("customer_id").size()
    icnt = df.groupby("product_id").size()
    keep_users = set(ucnt[ucnt >= min_user_reviews].index)
    keep_items = set(icnt[icnt >= min_item_reviews].index)
    df = df[df["customer_id"].isin(keep_users) & df["product_id"].isin(keep_items)]
    return df

def temporal_split(df: pd.DataFrame, test_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    cutoff = df["review_date"].max() - pd.Timedelta(days=test_days)
    train = df[df["review_date"] <= cutoff]
    test = df[df["review_date"] > cutoff]
    return train, test

@dataclass
class Mappings:
    user2idx: Dict[str, int]
    idx2user: List[str]
    item2idx: Dict[str, int]
    idx2item: List[str]

def build_mappings(users: pd.Series, items: pd.Series) -> Mappings:
    uniq_u = users.astype(str).unique().tolist()
    uniq_i = items.astype(str).unique().tolist()
    user2idx = {u: i for i, u in enumerate(uniq_u)}
    item2idx = {it: i for i, it in enumerate(uniq_i)}
    return Mappings(user2idx, uniq_u, item2idx, uniq_i)
