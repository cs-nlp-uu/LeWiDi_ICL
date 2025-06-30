# DeMeVa-UU at LeWiDi-2025

[TOC]

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

