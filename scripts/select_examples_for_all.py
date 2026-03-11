import argparse
import os
import sys
import json
import yaml
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np

# ---------------------------------------------------------------------------
# Resolve PROJECT_ROOT: prefer the environment variable, fall back to the
# parent directory of *this* script (i.e. the repository root).
# ---------------------------------------------------------------------------
_default_root = str(Path(__file__).resolve().parent.parent)
PROJECT_ROOT = os.getenv("PROJECT_ROOT", _default_root)
os.environ.setdefault("PROJECT_ROOT", PROJECT_ROOT)
project_root = Path(PROJECT_ROOT)
sys.path.append(str(project_root))

from src.load_data import load_data
from src.utils import select_examples, select_examples_by_labels

# Load config
with open(project_root / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)
for dataset_name in config['dataset_names']:
    for key, value in config['data'][dataset_name].items():
        config['data'][dataset_name][key] = value.replace('${PROJECT_ROOT}', str(project_root))

# model = SentenceTransformer('all-mpnet-base-v2')

# Helper: concatenate text fields for each dataset
def get_concat_text(dataset_name, text_dict):
    if dataset_name == 'CSC':
        return text_dict['context'] + ' [SEP] ' + text_dict['response']
    elif dataset_name == 'MP':
        return text_dict['post'] + ' [SEP] ' + text_dict['reply']
    elif dataset_name == 'Paraphrase':
        return text_dict['Question1'] + ' [SEP] ' + text_dict['Question2']
    elif dataset_name == 'VariErrNLI':
        return text_dict['context'] + ' [SEP] ' + text_dict['statement']
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

# Helper functions for embedding caching
def save_embeddings(embeddings, id2idx, out_path_prefix):
    np.save(f'{out_path_prefix}_embeddings.npy', embeddings)
    with open(f'{out_path_prefix}_id2idx.json', 'w') as f:
        json.dump(id2idx, f)

def load_embeddings(out_path_prefix):
    embeddings = np.load(f'{out_path_prefix}_embeddings.npy')
    with open(f'{out_path_prefix}_id2idx.json', 'r') as f:
        id2idx = json.load(f)
    return embeddings, id2idx

# def ensure_split_embeddings(dataset_name, split, ids, data, emb_dir):
#     out_prefix = emb_dir / f'{dataset_name}_{split}'
#     emb_file = f'{out_prefix}_embeddings.npy'
#     id2idx_file = f'{out_prefix}_id2idx.json'
#     if os.path.exists(emb_file) and os.path.exists(id2idx_file):
#         embeddings, id2idx = load_embeddings(out_prefix)
#     else:
#         texts = [get_concat_text(dataset_name, data[id_]['text']) for id_ in ids]
#         embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
#         id2idx = {id_: i for i, id_ in enumerate(ids)}
#         save_embeddings(embeddings, id2idx, out_prefix)
#     return embeddings, id2idx

# Main logic
def main(k = 15, seed=42):
    random.seed(seed)
    out_dir = project_root / 'examples'
    out_dir.mkdir(exist_ok=True)
    emb_dir = project_root / 'embeddings'
    emb_dir.mkdir(exist_ok=True)
    for dataset_name in config['dataset_names']:
        # if dataset_name != "MP":
        #     continue
        print(f'Processing {dataset_name}...')
        is_varierrnli = (dataset_name == 'VariErrNLI')
        # Load train split
        _, _, annotators_pe_train, ids_train, data_train = load_data(config['data'][dataset_name]['train_file'], is_varierrnli=is_varierrnli, is_test=False)
        # Build annotator->id set for train
        train_ann2ids = {}
        for idx, id_ in enumerate(ids_train):
            for ann in annotators_pe_train[idx]:
                if ann not in train_ann2ids:
                    train_ann2ids[ann] = set()
                train_ann2ids[ann].add(id_)
        # Ensure train embeddings cached (per split, not per annotator)
        # train_embeddings, train_id2idx = ensure_split_embeddings(dataset_name, 'train', ids_train, data_train, emb_dir)
        for split in ['dev', 'test']:
            out_path = out_dir / f'{dataset_name}_{split}_selected_{k}.json'
            if out_path.exists():
                continue
            if f'{split}_file' not in config['data'][dataset_name]:
                continue
            print(f'  Split: {split}')
            is_test = (split == 'test')
            _, _, annotators_pe, ids, data = load_data(config['data'][dataset_name][f'{split}_file'], is_varierrnli=is_varierrnli, is_test=is_test)
            # Ensure test/dev embeddings cached
            # test_embeddings, test_id2idx = ensure_split_embeddings(dataset_name, split, ids, data, emb_dir)
            result = {}
            for idx, id_ in enumerate(tqdm(ids)):
                entry = data[id_]
                # test_emb = test_embeddings[test_id2idx[id_]]
                for ann in annotators_pe[idx]:
                    # Only select from train entries annotated by this annotator
                    if ann not in train_ann2ids or len(train_ann2ids[ann]) == 0:
                        continue
                    ann_train_ids = list(train_ann2ids[ann])
                    # ann_train_embs = np.array([train_embeddings[train_id2idx[tid]] for tid in ann_train_ids])
                    # selected_ids = select_examples(test_emb, ann_train_embs, ann_train_ids, k=15)
                    selected_ids = select_examples_by_labels(
                        train_data=data_train, annotator_id=ann, annotator_train_ids=ann_train_ids, k=k
                    )
                    result[f'{id_}+{ann}'] = selected_ids
            with open(out_path, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f'    Saved: {out_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Pre-compute example selections for all datasets."
    )
    parser.add_argument("--k", type=int, default=15, help="Number of examples to select (default: 15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    main(k=args.k, seed=args.seed)