import os
import sys
import json
import yaml
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Add project root to sys.path
load_dotenv()
project_root = Path(os.getenv('PROJECT_ROOT'))
sys.path.append(str(project_root))

from src.load_data import load_data
from src.utils import select_examples

# Load config
with open(project_root / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)
for dataset_name in config['dataset_names']:
    for key, value in config['data'][dataset_name].items():
        config['data'][dataset_name][key] = value.replace('${PROJECT_ROOT}', str(project_root))

model = SentenceTransformer('all-mpnet-base-v2')

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

# Main logic
def main():
    out_dir = project_root / 'examples'
    out_dir.mkdir(exist_ok=True)
    for dataset_name in config['dataset_names']:
        print(f'Processing {dataset_name}...')
        is_varierrnli = (dataset_name == 'VariErrNLI')
        # Load train split
        _, _, annotators_pe_train, ids_train, data_train = load_data(config['data'][dataset_name]['train_file'], is_varierrnli=is_varierrnli, is_test=False)
        # Build annotator->id->concat_text for train
        train_ann2id2text = {}
        for idx, id_ in enumerate(ids_train):
            entry = data_train[id_]
            for ann in annotators_pe_train[idx]:
                if ann not in train_ann2id2text:
                    train_ann2id2text[ann] = {}
                train_ann2id2text[ann][id_] = get_concat_text(dataset_name, entry['text'])
        for split in ['dev', 'test']:
            out_path = out_dir / f'{dataset_name}_{split}_cosmrr_15.json'
            if out_path.exists():
                continue
            if f'{split}_file' not in config['data'][dataset_name]:
                continue
            print(f'  Split: {split}')
            is_test = (split == 'test')
            _, _, annotators_pe, ids, data = load_data(config['data'][dataset_name][f'{split}_file'], is_varierrnli=is_varierrnli, is_test=is_test)
            result = {}
            for idx, id_ in enumerate(tqdm(ids)):
                entry = data[id_]
                for ann in annotators_pe[idx]:
                    # Only select from train entries annotated by this annotator
                    if ann not in train_ann2id2text or len(train_ann2id2text[ann]) == 0:
                        continue
                    test_text = get_concat_text(dataset_name, entry['text'])
                    selected_ids = select_examples(model, test_text, train_ann2id2text[ann], k=15)
                    result[f'{id_}+{ann}'] = selected_ids
            with open(out_path, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f'    Saved: {out_path}')

if __name__ == '__main__':
    main() 