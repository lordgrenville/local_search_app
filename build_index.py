import logging
import faiss
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(message)s", datefmt='%H:%M')
d = 768
tokenizer = AutoTokenizer.from_pretrained("model_files")
model = AutoModel.from_pretrained("model_files")
device = torch.device("mps")
model.to(device)

def good_letter_number_ratio(s):
    num_letters = sum(map(lambda x: x.isalpha(), s))
    return num_letters / (len(s) - num_letters) > 0.5

def get_corpus():
    df = pd.read_csv('result.csv')
    df = df.loc[df['title'].ne('Table of Contents'), :]
    # df.body = df.body.fillna('').str.replace('\n', ' ').str.replace('\xa0', ' ')
    df['body'] = df['body'].fillna('')
    df['body'] = df['body'].replace({r'\r': ''}, regex=True)
    df['content'] = df['title'] + ' ' + df['body']
    df['body'] = df['body'].apply(lambda s: markdown2.markdown(s).strip())
    return df.loc[df['content'].apply(good_letter_number_ratio), :].reset_index(drop=True)

def get_embeddings(text_list):
    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return model_output.last_hidden_state[:, 0]

if __name__ == '__main__':
    index = faiss.IndexFlatL2(d)
    df = get_corpus()
    logging.debug('creating embeddings')
    embeddings = [get_embeddings(x).numpy(force=True) for x in df['title'] + ' ' + df['body']]
    index.add(np.stack(embeddings).reshape(len(df), d))
    faiss.write_index(index, 'latest.index')
    df[['title', 'body']].to_feather('latest.feather')
