import numpy as np
import argparse
import json
import os
from collections import defaultdict, Counter
from functools import reduce
from webanno_tsv import webanno_tsv_read_file, Document, Annotation
from typing import List, Union
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, performance_measure
from seqeval.scheme import IOB2
import seqeval


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


def to_char_bio(src_path: str, ref_path: str) -> List[List[str]]:
    ref_doc = webanno_tsv_read_file(ref_path)
    # Parse the WebAnno TSV file
    doc = webanno_tsv_read_file(src_path)
    # Initialize a list to store character-level BIO tags
    bio_tags_list = []
    for target_label in LABELS:
        bio_tags = ['#'] * len(ref_doc.text)  # Default to '#' for all characters
        # Pick interested sentences and default them to 'O'
        for annotation in ref_doc.annotations:
            label = annotation.label
            if label != target_label:
                continue
            sentences = ref_doc.annotation_sentences(annotation)
            for sentence in sentences:
                tokens = ref_doc.sentence_tokens(sentence)
                start_char, end_char = tokens[0].start, tokens[-1].end
                bio_tags[start_char:end_char] = ['O'] * (end_char-start_char)

        for annotation in doc.annotations:
            label = annotation.label
            if label != target_label:
                continue

            start_token, end_token = annotation.tokens[0], annotation.tokens[-1]
            start_char = start_token.start
            end_char = end_token.end
            # Sanity check
            if ref_doc.text[start_char:end_char] != annotation.text:
                msg = f"ERROR: src: {src_path}, annotated '{annotation.text}', text: '{ref_doc.text[start_char:end_char]}'"
                print(msg)

            if 'I-' in bio_tags[start_char]:
                # Overlapping, it's annotated by another annotations, we connect them as one annotations
                pass
            else:
                if bio_tags[start_char] != '#':
                    # Assign BIO tags to characters in the entity span
                    bio_tags[start_char] = f'B-{label}'  # Beginning of the entity

            for i in range(start_char + 1, end_char):
                if bio_tags[i] != '#':
                    bio_tags[i] = f'I-{label}'  # Inside the entity

        # Remove unannotated sentences from bio list.
        # bio_tags = [x for x in filter(lambda x: x != '#', bio_tags)]
        for idx, tag in enumerate(bio_tags):
            if tag == '#':
                bio_tags[idx] = 'O'

        bio_tags_list.append(bio_tags)

    return bio_tags_list


def get_parse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--reference_dir', type=str, help='Path to the reference data, e.g. training/validation/test data', required=True)
    parser.add_argument('--prediction_dir', type=str, help='Path to save the prediction results', required=True)
    parser.add_argument('--score_dir', type=str, help='Path to store scores', default='./results/scores_em')
    parser.add_argument('--average', type=str, choices=['macro', 'micro', 'weighted'],
                       help='Type of averaging for metrics calculation', default='macro')
    return parser


# def calc_score(all_ref_bio_tags_list, all_pred_bio_tags_list, pred_dir):
#     scores = {}
#     # f1 = f1_score(ref_bio_tags_list, pred_bio_tags_list, average=average_type)
#     # precision = precision_score(ref_bio_tags_list, pred_bio_tags_list, average=average_type)
#     # recall = recall_score(ref_bio_tags_list, pred_bio_tags_list, average=average_type)
#     # scores[f"overall_{average_type}_precision"] = precision
#     # scores[f"overall_{average_type}_recall"] = recall
#     # scores[f"overall_{average_type}_f1"] = f1
#     # report = classification_report(ref_bio_tags_list, pred_bio_tags_list, output_dict=True, scheme=IOB2, mode='strict')
#     # # Add overall scores based on the specified average type
#     # for average_type in ['macro', 'micro', 'weighted']:
#     #     scores[f"overall_{average_type}_precision"] = report[f"{average_type} avg"]["precision"] * 100
#     #     scores[f"overall_{average_type}_recall"] = report[f"{average_type} avg"]["recall"] * 100
#     #     scores[f"overall_{average_type}_f1"] = report[f"{average_type} avg"]["f1-score"] * 100

#     # print("Scores:\n", json.dumps(scores, indent=2))

#     ################################################################################
#     # For each class
#     ################################################################################
#     label_to_ref_bio_tags_list = defaultdict(list)
#     label_to_pred_bio_tags_list = defaultdict(list)
#     for ref_bio_tags_list, pred_bio_tags_list in zip(all_ref_bio_tags_list, all_pred_bio_tags_list):
#         if len(ref_bio_tags_list) != len(LABELS):
#             print('ERROR: ref bio tags list')
#         if len(pred_bio_tags_list) != len(LABELS):
#             print('ERROR: pred bio tags list')

#         for label, ref_bio_tags, pred_bio_tags in zip(LABELS, ref_bio_tags_list, pred_bio_tags_list):
#             label_to_ref_bio_tags_list[label].extend(ref_bio_tags)
#             label_to_pred_bio_tags_list[label].extend(pred_bio_tags)
#             if len(label_to_ref_bio_tags_list[label]) != len(label_to_pred_bio_tags_list[label]):
#                 print('ERROR: label_to_ref_pred_bio_tags')


#     for label in label_to_ref_bio_tags_list.keys():
#         ref_bio_tags_list = label_to_ref_bio_tags_list[label]
#         pred_bio_tags_list = label_to_pred_bio_tags_list[label]
#         # accuracy = accuracy_score(ref_bio_tags_list, pred_bio_tags_list)
#         # Calculate scores using the specified average type
#         # f1 = f1_score(ref_bio_tags_list, pred_bio_tags_list, average=average_type)
#         # precision = precision_score(ref_bio_tags_list, pred_bio_tags_list, average=average_type)
#         # recall = recall_score(ref_bio_tags_list, pred_bio_tags_list, average=average_type)
#         # scores[f"{label}_{average_type}_precision"] = precision
#         # scores[f"{label}_{average_type}_recall"] = recall
#         # scores[f"{label}_{average_type}_f1"] = f1
#         import ipdb; ipdb.set_trace()
    # print("Scores:\n", json.dumps(scores, indent=2))

    # with open(os.path.join(pred_dir, f'00scores_{average_type}.json'), 'w') as fd:
    #     json.dump(scores, fd, indent=2)


if __name__ == "__main__":
    parser = get_parse()
    args = parser.parse_args()

    ref_dir = args.reference_dir
    pred_dir = args.prediction_dir
    score_dir = args.score_dir
    average_type = args.average  # Get the average type from command line

    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(score_dir, exist_ok=True)

    ref_file_names = sorted([fp for fp in os.listdir(ref_dir) if os.path.isfile(f'{ref_dir}/{fp}') and fp.endswith('.tsv')])

    if len(ref_file_names) == 0:
        raise Exception("ERROR: No reference files found, configuration error?")
    all_ref_bio_tags_list = []
    for ref_file_name in ref_file_names:
        src_path = os.path.join(ref_dir, ref_file_name)
        ref_path = src_path
        all_ref_bio_tags_list.append(to_char_bio(src_path, ref_path))

    pred_file_names = sorted([fp for fp in os.listdir(pred_dir) if os.path.isfile(f'{pred_dir}/{fp}') and fp.endswith('.tsv')])
    all_pred_bio_tags_list = []
    for idx, ref_file_name in enumerate(ref_file_names):
        try:
            src_path = os.path.join(pred_dir, ref_file_name)
            ref_path = os.path.join(ref_dir, ref_file_name)
            all_pred_bio_tags_list.append(to_char_bio(src_path, ref_path))
        except FileNotFoundError:
            nbr_labels = len(all_ref_bio_tags_list[idx])
            assert nbr_labels == len(LABELS), "ERROR: reference tags doesn't have ${len(LABELS)} labels."
            pred = []
            for label_idx in range(nbr_labels):
                pred.append(['O'] * len(all_ref_bio_tags_list[idx][label_idx]))

            print(f"WARN: {ref_file_name} is missing, fill 'O' list as default prediction")
            all_pred_bio_tags_list.append(pred)
    # Sanity checking
    for idx, (ref_list, pred_list) in enumerate(zip(all_ref_bio_tags_list, all_pred_bio_tags_list)):
        for label_idx, (ref, pred) in enumerate(zip(ref_list, pred_list)):
            assert len(ref) == len(pred), f'ERROR: {ref_file_names[idx]}, label: {LABELS[label_idx]}, reference length: {len(ref)}, prediction length: {len(pred)}'


    ################################################################################
    # Evaluation: merge labels into full sequences and use classification_report
    ################################################################################
    # Transpose: regroup by document
    # ref_bio_tags_list = flatten(flatten(all_ref_bio_tags_list))
    # pred_bio_tags_list = flatten(flatten(all_pred_bio_tags_list))
    # for average_type in ["macro", "micro", "weighted"]:
    ################################################################################
    # For each class
    ################################################################################
    label_to_ref_bio_tags_list = defaultdict(list)
    label_to_pred_bio_tags_list = defaultdict(list)
    for ref_bio_tags_list, pred_bio_tags_list in zip(all_ref_bio_tags_list, all_pred_bio_tags_list):
        if len(ref_bio_tags_list) != len(LABELS):
            print('ERROR: ref bio tags list')
        if len(pred_bio_tags_list) != len(LABELS):
            print('ERROR: pred bio tags list')

        for label, ref_bio_tags, pred_bio_tags in zip(LABELS, ref_bio_tags_list, pred_bio_tags_list):
            label_to_ref_bio_tags_list[label].extend(ref_bio_tags)
            label_to_pred_bio_tags_list[label].extend(pred_bio_tags)
            if len(label_to_ref_bio_tags_list[label]) != len(label_to_pred_bio_tags_list[label]):
                print('ERROR: label_to_ref_pred_bio_tags')


    for label in label_to_ref_bio_tags_list.keys():
        ref_bio_tags_list = label_to_ref_bio_tags_list[label]
        pred_bio_tags_list = label_to_pred_bio_tags_list[label]
        # accuracy = accuracy_score(ref_bio_tags_list, pred_bio_tags_list)
        # Calculate scores using the specified average type
        # f1 = f1_score(ref_bio_tags_list, pred_bio_tags_list, average=average_type)
        # precision = precision_score(ref_bio_tags_list, pred_bio_tags_list, average=average_type)
        # recall = recall_score(ref_bio_tags_list, pred_bio_tags_list, average=average_type)
        # scores[f"{label}_{average_type}_precision"] = precision
        # scores[f"{label}_{average_type}_recall"] = recall
        # scores[f"{label}_{average_type}_f1"] = f1

        # ref_bio_tags_list = [ref_bio_tags_list]
        # pred_bio_tags_list = [pred_bio_tags_list]
        # # report = classification_report(ref_bio_tags_list, pred_bio_tags_list, output_dict=True, mode='default', scheme=IOB2)
        # report = classification_report(ref_bio_tags_list, pred_bio_tags_list, output_dict=False, mode='default', scheme=IOB2)
        # # scores = {}
        # print(f'{label}\n', report)
        # ref = np.array(ref_bio_tags_list)
        # pred = np.array(pred_bio_tags_list)
        # equal = ref == pred

        # for average_type in ['macro', 'micro', 'weighted']:
        #     scores[f"{label}_{average_type}_precision"] = report[f"{average_type} avg"]["precision"] * 100
        #     scores[f"{label}_{average_type}_recall"] = report[f"{average_type} avg"]["recall"] * 100
        #     scores[f"{label}_{average_type}_f1"] = report[f"{average_type} avg"]["f1-score"] * 100

        from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, performance_measure
        # for average in ['macro', 'micro', 'weighted']:
        for mode in ['default', 'strict']:
            for average in ['macro', 'micro', 'weighted']:
                f1 = f1_score([ref_bio_tags_list], [pred_bio_tags_list], average=average, mode=mode)
                precision = precision_score([ref_bio_tags_list], [pred_bio_tags_list], average=average, mode=mode)
                recall = recall_score([ref_bio_tags_list], [pred_bio_tags_list], average=average, mode=mode)
                print(f"{mode} {average} - {label} - Precision: {precision * 100}%")
                print(f"{mode} {average} - {label} - Recall: {recall * 100}%")
                print(f"{mode} {average} - {label} - F1: {f1 * 100}%")
                pm = performance_measure([ref_bio_tags_list], [pred_bio_tags_list])
                print('================================================================================')
                # print(pm)
                # if label == 'PROGLANG':
                #     import ipdb; ipdb.set_trace()
                # if precision == 0:
                #     import ipdb; ipdb.set_trace()
        print('--------------------------------------------------------------------------------')
        # y_true = [['O', 'O', 'B-MISC', 'I-MISC', 'B-MISC', 'O', 'O', 'B-MISC', 'I-MISC', 'O']]
        # y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'B-MISC', 'I-MISC', 'O', 'B-MISC', 'I-MISC', 'O']]
        # x = performance_measure(y_true, y_pred)
        # print(x)
        # x = f1_score(y_true, y_pred, average='micro')
        # print(x)
        # x = f1_score(y_true, y_pred,  average='macro')
        # print(x)
        # x = f1_score(y_true, y_pred,  average='weighted')
        # print(x)
        # from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
        # y_true = [['O', 'O', 'B-MISC', 'B-MISC', 'B-MISC', 'O', 'O', 'B-MISC', 'B-MISC', 'O']]
        # y_pred = [['O', 'O', 'B-MISC', 'B-MISC', 'B-MISC', 'B-MISC', 'O', 'B-MISC', 'B-MISC', 'O']]
        # y = confusion_matrix(y_true[0], y_pred[0]).ravel()
        # print(y)
        # y = f1_score(y_true[0], y_pred[0], average='micro')
        # print(y)
        # y = f1_score(y_true[0], y_pred[0], average= 'macro')
        # print(y)
        # y = f1_score(y_true[0], y_pred[0],  average='weighted')
        # print(y)


    # all_ref_grouped = list(zip(*[label_seq for label_seq in all_ref_bio_tags_list]))
    # all_pred_grouped = list(zip(*[label_seq for label_seq in all_pred_bio_tags_list]))

    # # Rebuild full label sequence per document
    # merged_ref = [flatten(doc_labels) for doc_labels in all_ref_grouped]
    # merged_pred = [flatten(doc_labels) for doc_labels in all_pred_grouped]

    # # Compute full classification report with the specified average type
    # report = classification_report(merged_ref, merged_pred, output_dict=True, mode='strict', scheme=IOB2)

    # # Extract individual scores
    # scores = {}
    # for label in LABELS:
    #     if label in report:
    #         scores[f"{label}_precision"] = report[label]["precision"]
    #         scores[f"{label}_recall"] = report[label]["recall"]
    #         scores[f"{label}_f1"] = report[label]["f1-score"]
    #     else:
    #         scores[f"{label}_precision"] = 0.0
    #         scores[f"{label}_recall"] = 0.0
    #         scores[f"{label}_f1"] = 0.0

    # # Add overall scores based on the specified average type
    # scores[f"overall_{average_type}_precision"] = report[f"{average_type} avg"]["precision"]
    # scores[f"overall_{average_type}_recall"] = report[f"{average_type} avg"]["recall"]
    # scores[f"overall_{average_type}_f1"] = report[f"{average_type} avg"]["f1-score"]

    # print("Scores:\n", json.dumps(scores, indent=2))

    # with open(os.path.join(score_dir, f'scores_{average_type}.json'), 'w') as fd:
    #     json.dump(scores, fd, indent=2)
