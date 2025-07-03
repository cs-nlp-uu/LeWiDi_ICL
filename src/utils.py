import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

def select_examples(model, test_entry, train_entries, k=15, lamb=0.7):
    train_ids = list(train_entries.keys())
    train_texts = list(train_entries.values())
    train_vecs  = model.encode(train_texts, normalize_embeddings=True)
    q_vec = model.encode(test_entry, normalize_embeddings=True)
    sim   = (train_vecs * q_vec).sum(axis=1)
    chosen, candidate_idx = [], list(range(len(train_entries)))
    while len(chosen) < k and candidate_idx:
        if not chosen:
            idx = int(np.argmax(sim))
        else:
            cand_vecs   = train_vecs[candidate_idx]
            div         = cosine_similarity(cand_vecs, train_vecs[chosen]).max(axis=1)
            mmr_scores  = lamb * sim[candidate_idx] - (1 - lamb) * div
            idx = candidate_idx[int(np.argmax(mmr_scores))]
        chosen.append(idx)
        candidate_idx.remove(idx)
    return [train_ids[i] for i in chosen]

# model = sentence_transformers.SentenceTransformer('all-mpnet-base-v2')