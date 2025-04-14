import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from data_processing import *
from sentence_transformers import SentenceTransformer
def find_perfect_matches( similarity_threshold=0.99):
    ref = pd.read_csv('data/reference_rooms.csv')
    upc = pd.read_csv('data/updated_core_rooms.csv')
    reference_rooms, supplier_rooms = main(ref, upc)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    ref_s = reference_rooms.iloc[0:20000]
    supplier_s = supplier_rooms.iloc[0:30000]
    matches = enhanced_room_matching(ref_s, supplier_s, model, similarity_threshold=similarity_threshold, feature_weight=0.3)
    matches = matches.drop_duplicates(subset=['reference_room_name','supplier_room_name'])
    matches['text_similarity'] = matches['text_similarity'].round(2)
    perfect_matches = matches[matches['text_similarity']>0.99].sample(250)
    top_similarity = matches[(matches['text_similarity']>=0.96) & (matches['text_similarity']<=0.98)].sample(500)
    low_similarity = matches[(matches['text_similarity']>=0.85) & (matches['text_similarity']<0.88)].sample(250)

    print(len(perfect_matches), len(top_similarity), len(low_similarity))
    label_data = pd.concat([perfect_matches, top_similarity, low_similarity])
    return label_data

