import yaml
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss

from recsys.explain import top_phrases

class RecRequest(BaseModel):
    customer_id: str
    top_k: int | None = None

app = FastAPI(title="Review-based Recommender")

cfg = yaml.safe_load(open("configs/config.yaml"))

# Load artifacts
user_docs = pd.read_parquet(f"{cfg['data']['processed_dir']}/user_docs.parquet")
ud = dict(zip(user_docs["customer_id"].astype(str), user_docs["user_doc"]))

item_docs = pd.read_parquet(f"{cfg['data']['processed_dir']}/item_docs.parquet")
idoc = dict(zip(item_docs["product_id"].astype(str), item_docs["item_doc"]))

item_ids = pd.read_parquet("artifacts/index/item_ids.parquet")["product_id"].astype(str).tolist()

index = faiss.read_index("artifacts/index/items.faiss")
# optionally move to GPU at serve time
if cfg["index"]["use_gpu"]:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)

model = SentenceTransformer("artifacts/models/two_tower", device="cuda")

@app.get("/health")
def health():
    return {"status": "ok", "users": len(ud), "items": len(item_ids)}

@app.post("/recommend")
def recommend(req: RecRequest):
    if req.customer_id not in ud:
        raise HTTPException(status_code=404, detail="Unknown customer_id (not in training docs).")
    k = req.top_k or cfg["serve"]["top_k"]

    uemb = model.encode([ud[req.customer_id]], batch_size=1, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(uemb, k)

    recs = []
    for rank, idx in enumerate(I[0].tolist(), start=1):
        pid = item_ids[idx]
        phrases = top_phrases(ud[req.customer_id], idoc.get(pid, ""), top_n=3)
        recs.append({
            "rank": rank,
            "product_id": pid,
            "score": float(D[0][rank-1]),
            "because": phrases
        })
    return {"customer_id": req.customer_id, "recommendations": recs}
