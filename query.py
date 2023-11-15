import faiss
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
from build_index import get_embeddings

d = 768
model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
device = torch.device("mps")
model.to(device)
df = pd.read_feather('latest.feather')
index = faiss.read_index('latest.index')

def query_joplin(search_string, n=40):
    xq = get_embeddings(search_string).detach().cpu().numpy()[0].reshape(1, -1)
    _, indices = index.search(xq, n)
    return df.loc[indices[0], ['title', 'body']]
