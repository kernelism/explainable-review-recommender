import os
import yaml
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    pdir = cfg["data"]["processed_dir"]

    item_docs = pd.read_parquet(f"{pdir}/item_docs.parquet")
    id_list = item_docs["product_id"].astype(str).tolist()
    texts = item_docs["item_doc"].tolist()

    model = SentenceTransformer("artifacts/models/two_tower", device="cuda")
    emb = model.encode(texts, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True).astype("float32")
    d = emb.shape[1]

    # Exact baseline index: IndexFlatIP (inner product). FAISS notes flat indexes guarantee exact results. [web:35]
    index = faiss.IndexFlatIP(d)

    if cfg["index"]["use_gpu"]:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(emb)

    os.makedirs("artifacts/index", exist_ok=True)
    # FAISS GPU index must be moved back to CPU for saving
    index_cpu = faiss.index_gpu_to_cpu(index) if cfg["index"]["use_gpu"] else index
    faiss.write_index(index_cpu, "artifacts/index/items.faiss")

    pd.DataFrame({"product_id": id_list}).to_parquet("artifacts/index/item_ids.parquet", index=False)
    np.save("artifacts/embeddings/item_emb.npy", emb)
    print(f"Saved index with {len(id_list)} items (dim={d})")

if __name__ == "__main__":
    main()
