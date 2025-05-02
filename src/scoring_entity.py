import argparse
import json
import os
from collections import defaultdict
from functools import reduce
from webanno_tsv import webanno_tsv_read_file, Document, Annotation
from typing import List, Union
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


LABELS = [
    'CONFERENCE',
    'DATASET',
    'EVALMETRIC',
    'LICENSE',
    'ONTOLOGY',
    'PROGLANG',
    'PROJECT',
    'PUBLICATION',
    'SOFTWARE',
    'WORKSHOP'
]


def flatten(lst):
    return reduce(lambda x, y: x + y, lst)


def to_char_bio_entity(tsv_path):
    ref_doc = webanno_tsv_read_file(tsv_path)
    # Parse the WebAnno TSV file

    # Map each token to its index for quick lookup
    token_to_index = {token: idx for idx, token in enumerate(ref_doc.tokens)}

    # Initialize the list to store BIO tag sequences for each label
    bio_tags_list = []

    # Create a separate BIO tag sequence for each label
    for target_label in LABELS:
        # Start with an all-'O' tag list
        bio_tags = ['O'] * len(ref_doc.tokens)

        # Iterate over annotations to apply tags for the current label
        for annotation in ref_doc.annotations:
            if annotation.label == target_label:
                entity_tokens = annotation.tokens
                for i, token in enumerate(entity_tokens):
                    token_idx = token_to_index[token]
                    prefix = 'B-' if i == 0 else 'I-'
                    bio_tags[token_idx] = f'{prefix}{target_label}'

        bio_tags_list.append(bio_tags)

    return bio_tags_list


def get_parse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--reference_dir', type=str, help='Path to the reference data, e.g. training/validation/test data', required=True)
    parser.add_argument('--prediction_dir', type=str, help='Path to save the prediction results', required=True)
    parser.add_argument('--score_dir', type=str, help='Path to store scores', default='./results/scores_em')
    return parser


if __name__ == "__main__":
    parser = get_parse()
    args = parser.parse_args()

    ref_dir = args.reference_dir
    pred_dir = args.prediction_dir
    score_dir = args.score_dir

    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(score_dir, exist_ok=True)

    ref_file_names = sorted([fp for fp in os.listdir(ref_dir) if os.path.isfile(f'{ref_dir}/{fp}') and fp.endswith('.tsv')])
    pred_file_names = sorted([fp for fp in os.listdir(pred_dir) if os.path.isfile(f'{pred_dir}/{fp}') and fp.endswith('.tsv')])

    if len(ref_file_names) == 0:
        raise Exception("ERROR: No reference files found, configuration error?")

    all_ref_bio_tags_list = []
    for ref_file_name in ref_file_names:
        ref_path = os.path.join(ref_dir, ref_file_name)
        all_ref_bio_tags_list.append(to_char_bio_entity(ref_path))

    all_pred_bio_tags_list = []
    for pred_file_name in pred_file_names:
        src_path = os.path.join(pred_dir, pred_file_name)
        all_pred_bio_tags_list.append(to_char_bio_entity(src_path))
    # Sanity checking
    for idx, (ref_list, pred_list) in enumerate(zip(all_ref_bio_tags_list, all_pred_bio_tags_list)):
        for label_idx, (ref, pred) in enumerate(zip(ref_list, pred_list)):
            assert len(ref) == len(pred), f'ERROR: {ref_file_names[idx]}, label: {LABELS[label_idx]}, reference length: {len(ref)}, prediction length: {len(pred)}'


    ################################################################################
    # Evaluation: merge labels into full sequences and use classification_report
    ################################################################################
    # Transpose: regroup by document
    all_ref_grouped = list(zip(*[label_seq for label_seq in all_ref_bio_tags_list]))
    all_pred_grouped = list(zip(*[label_seq for label_seq in all_pred_bio_tags_list]))

    # Rebuild full label sequence per document
    merged_ref = [flatten(doc_labels) for doc_labels in all_ref_grouped]
    merged_pred = [flatten(doc_labels) for doc_labels in all_pred_grouped]

    # Compute full classification report
    report = classification_report(merged_ref, merged_pred, output_dict=True)

    # Extract individual scores
    scores = {}
    for label in LABELS:
        if label in report:
            scores[f"{label}_precision"] = report[label]["precision"]
            scores[f"{label}_recall"] = report[label]["recall"]
            scores[f"{label}_f1"] = report[label]["f1-score"]
        else:
            scores[f"{label}_precision"] = 0.0
            scores[f"{label}_recall"] = 0.0
            scores[f"{label}_f1"] = 0.0

    # Add overall scores (micro/macro averages)
    if "micro avg" in report:
        scores["overall_micro_precision"] = report["micro avg"]["precision"]
        scores["overall_micro_recall"] = report["micro avg"]["recall"]
        scores["overall_micro_f1"] = report["micro avg"]["f1-score"]
    
    if "macro avg" in report:
        scores["overall_macro_precision"] = report["macro avg"]["precision"]
        scores["overall_macro_recall"] = report["macro avg"]["recall"]
        scores["overall_macro_f1"] = report["macro avg"]["f1-score"]
    
    if "weighted avg" in report:
        scores["overall_weighted_precision"] = report["weighted avg"]["precision"]
        scores["overall_weighted_recall"] = report["weighted avg"]["recall"]
        scores["overall_weighted_f1"] = report["weighted avg"]["f1-score"]

    print("Scores:\n", json.dumps(scores, indent=2))

    with open(os.path.join(score_dir, 'scores.json'), 'w') as fd:
        json.dump(scores, fd, indent=2)