# DeMeVa-UU at LeWiDi-2025

In-Context Learning (ICL) system for the **Le**arning **Wi**th **Di**sagreements 2025 shared task.
The system predicts per-annotator labels for four annotation tasks by prompting large language models with personalised in-context examples.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Running Experiments](#running-experiments)
  - [1. Pre-compute Example Selections](#1-pre-compute-example-selections)
  - [2. Run ICL Predictions](#2-run-icl-predictions)
  - [3. Submit Batch Requests (optional)](#3-submit-batch-requests-optional)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Annotators](#annotators)

## Prerequisites

- Python 3.9 or later
- An OpenAI-compatible API key (set as `OPENAI_API_KEY` environment variable)

## Installation

```bash
git clone https://github.com/cs-nlp-uu/LeWiDi_ICL.git
cd LeWiDi_ICL
pip install -r requirements.txt
```

## Data Setup

Place the LeWiDi-2025 data files under `data/data_evaluation_phase/` so the
directory tree looks like:

```
data/data_evaluation_phase/
├── CSC/
│   ├── CSC_train.json
│   ├── CSC_dev.json
│   └── CSC_test_clear.json
├── MP/
│   ├── MP_train.json
│   ├── MP_dev.json
│   └── MP_test_clear.json
├── Paraphrase/
│   ├── Paraphrase_train.json
│   ├── Paraphrase_dev.json
│   └── Paraphrase_test_clear.json
└── VariErrNLI/
    ├── VariErrNLI_train.json
    ├── VariErrNLI_dev.json
    └── VariErrNLI_test_clear.json
```

The `data/` directory is listed in `.gitignore` and is **not** checked into version control.

## Running Experiments

### 1. Pre-compute Example Selections

Before running ICL predictions with the `uniform` selection method you need
to pre-compute example sets.  Pre-computed files for `cosmrr` (used by
`topk`) are already included in `examples/`.

```bash
python scripts/select_examples_for_all.py          # defaults: k=15, seed=42
python scripts/select_examples_for_all.py --k 10   # choose a different k
```

### 2. Run ICL Predictions

```bash
export OPENAI_API_KEY="sk-..."

# Use defaults from config.yaml
python src/run_code.py

# Override any setting via CLI flags
python src/run_code.py --model gpt-4o --n-shots 10 --selection-method uniform \
    --test-mode dev --datasets CSC MP
```

Run `python src/run_code.py --help` for the full list of options.
All flags fall back to the values in `config.yaml` when not specified.

Predictions are saved to `predictions/` and run logs to `logs/`.

### 3. Submit Batch Requests (optional)

For large-scale runs you can use the OpenAI Batch API:

```bash
export OPENAI_API_KEY="sk-..."
python src/api.py --model gpt-4o
```

## Configuration

Experiment defaults live in **`config.yaml`**:

| Key                | Description                                  | Default                        |
|--------------------|----------------------------------------------|--------------------------------|
| `model`            | LLM model name                               | `gpt-4o`                       |
| `base_url`         | API endpoint                                 | `https://api.openai.com/v1`    |
| `n_shots`          | Number of in-context examples                | `10`                           |
| `selection_method` | Example selection (`random`/`topk`/`uniform`) | `uniform`                      |
| `test_mode`        | Evaluation split (`dev`/`test`)              | `test`                         |
| `n_entry`          | Entries to process (`-1` = all)              | `-1`                           |
| `random_seed`      | Seed for reproducibility                     | `42`                           |

Environment variables `OPENAI_API_KEY` and `OPENAI_BASE_URL` are read
automatically; CLI flags take the highest priority.

## Project Structure

```
├── config.yaml                 # Experiment configuration
├── requirements.txt            # Python dependencies
├── prompts/
│   └── prompt_template.json    # Prompt templates (versioned)
├── src/
│   ├── run_code.py             # Main entry point – ICL predictions
│   ├── api.py                  # OpenAI Batch API helper
│   ├── load_data.py            # Data loading utilities
│   ├── utils.py                # Example selection (MMR, stratified)
│   └── evaluation.py           # Evaluation metrics
├── scripts/
│   └── select_examples_for_all.py  # Pre-compute example selections
├── examples/                   # Pre-computed example ID files
├── embeddings/                 # Cached embeddings
└── notebooks/                  # Jupyter notebooks for exploration
```

## Datasets

Four datasets are used in this task:
1. Conversational Sarcasm Corpus (CSC)
2. MULtiPico dataset (MP)
3. Paraphrase Detection dataset (Par)
4. VarErr NLI dataset (VariErrNLI)


### Format

Example:

```json
{
  "annotation task": "sarcasm detection",
  "text": {
    "context": "You were watching Steve's new puppy for a week while he was traveling in Italy. The puppy made a total mess out of your apartment.  Clearly, Steve did not train the puppy properly.  Steve texts you and says, \"so, how is my puppy doing?\"",
    "response": "okay, very cute if a little messy!"
  },
  "number of annotators": 6,
  "annotators": "Ann288,Ann289,Ann290,Ann291,Ann292,Ann293",
  "number of annotations": 6,
  "annotations": {
    "Ann288": "2",
    "Ann289": "5",
    "Ann290": "3",
    "Ann291": "3",
    "Ann292": "2",
    "Ann293": "6"
  },
  "soft_label": {
    "0": 0.0,
    "1": 0.0,
    "2": 0.3333333333333333,
    "3": 0.3333333333333333,
    "4": 0.0,
    "5": 0.16666666666666666,
    "6": 0.16666666666666666
  },
  "split": "dev",
  "lang": "en",
  "other_info": {
    "context+speaker": "121_1049"
  }
}
```

Attention:

- `annotation task`: one of `sarcasm detection`, `irony detection`, `paraphrase detection`, and `natural language inference`.

- `text` and `other-info`: task-specific values
  
### Examples

1. CSC:

```json
{
        "annotation task": "sarcasm detection",
        "text": {
            "context": "You were watching Steve's new puppy for a week while he was traveling in Italy. The puppy made a total mess out of your apartment.  Clearly, Steve did not train the puppy properly.  Steve texts you and says, \"so, how is my puppy doing?\"",
            "response": "okay, very cute if a little messy!"
        },
        ...
        "annotations": {
            "Ann288": "2",
            " Ann289": "5",
            " Ann290": "3",
            " Ann291": "3",
            " Ann292": "2",
            " Ann293": "6"
        },
        "soft_label": {
            "0": 0.0,
            "1": 0.0,
            "2": 0.3333333333333333,
            "3": 0.3333333333333333,
            "4": 0.0,
            "5": 0.16666666666666666,
            "6": 0.16666666666666666
        },
        ...
        "other_info": {
            "context+speaker": "121_1049"
        }
    }
```

2. MP

```json
{
        "annotation task": "irony detection",
        "text": {
            "post": "Last Saturday night was homecoming and today  every single kid in my son's friend group tested positive for Influenza....so yeah..now is a good time to get your flu shot.",
            "reply": "@USER Oh no!!"
        },
        ...
        "annotations": {
            "Ann0": "0",
            "Ann10": "0",
            "Ann16": "0"
        },
        "soft_label": {
            "0.0": 1.0,
            "1.0": 0
        },
        ...
        "other_info": {
            "source": "twitter",
            "level": 1.0,
            "language_variety": "us"
        }
    }
```

3. Par

```json
{
        "annotation task": "paraphrase detection",
        "text": {
            "Question1": "What are the chances for a high-school student who didn't do any extracurricular activities until his last year to get accepted to a top US school?",
            "Question2": "Is a student who sucked in High School but ended up being a top PhD student in a top school a misunderstood genius?"
        },
        ...
        "annotations": {
            "Ann1": "-3",
            "Ann2": "-4",
            "Ann3": "0",
            "Ann4": "-4"
        },
        "soft_label": {
            "-5": 0.0,
            "-4": 0.5,
            "-3": 0.25,
            "-2": 0.0,
            "-1": 0.0,
            "0": 0.25,
            "1": 0.0,
            "2": 0.0,
            "3": 0.0,
            "4": 0.0,
            "5": 0.0
        },
        ...
        "other_info": {
            "explanations": [
                "Q1 asks about the chances of less well-performing student being accepted to top Uni, whereas Q2 asks about if a kind of students can be a misunderstood genius",
                " different topics",
                "term overlap \"top school\", Q1 specifies named entity \"US\", Q2: \"high school\". They refer to different concepts. ",
                "The main sense of the two sentences is different."
            ]
        }
    }
```

4. VarErrNLI

```json
{
        "annotation task": "natural language inference",
        "text": {
            "context": "Part of the reason for the difference in pieces per possible delivery may be due to the fact that five percent of possible residential deliveries are businesses, and it is thought, but not known, that a lesser percentage of possible deliveries on rural routes are businesses.",
            "statement": "It is thought, but not known, that a lesser percentage of possible deliveries on rural routes are businesses, and part of the reason for the difference in pieces per possible delivery, may be due to the fact that five percent of possible residential deliveries are businesses."
        },
        ...
        "annotations": {
            "Ann1": "contradiction",
            "Ann3": "entailment"
        },
        "soft_label": {
            "contradiction": {
                "0": 0.5,
                "1": 0.5
            },
            "entailment": {
                "0": 0.5,
                "1": 0.5
            },
            "neutral": {
                "0": 1.0,
                "1": 0.0
            }
        },
        ...
        "other_info": {
            "explanations": [
                "The reason for the diffenrence in pieces per possible delivery mentioned in the context is that the difference percentage of businesses deliveries on residential and rural routes. But the reason in the statement only include the percentage of residential deliveries, not the diffenrence of deliveries.",
                "Statement just changed the order of two hypothesis in the context."
            ]
        }
    }
```

### Annotators

Annatator metadata is stores in `${DATASET_NAME}_annotators_meta.json` files. These files contain the following fields:

1. All datasets: `annotator_id`, `age`, `gender`
2. MP, Par, VariErrNLI: `nationality`
3. Par, VariErrNLI: `education`
4. MP: `ethnicity simplified`, `country of birth`, `country of residence`, `student status`, `employment status`.

