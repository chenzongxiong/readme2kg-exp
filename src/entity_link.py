# Standard library
import time
import json
import logging
import math
import multiprocessing as mp
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple
from dataclasses import dataclass

# Third-party
import numpy as np
import pandas as pd
from termcolor import colored
from tqdm import tqdm
import requests
# from huggingface_hub import list_datasets
from webanno_tsv import webanno_tsv_read_file, Document, Annotation, Token
from base_predictor import LABELS
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class Config:
    threshold: float = 0.7
    topk: int = 10


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def extract_entity_from_tsv(files: List[Path], target_label: str):
    mentions = []

    for file in files:
        doc = webanno_tsv_read_file(file)
        for annotation in doc.annotations:
            if annotation.label != target_label:
                continue

            sentences = doc.annotation_sentences(annotation)
            for sentence in sentences:
                mentions.append({
                    "sentence": sentence.text,
                    "entity_text": annotation.text,
                    "entity_type": annotation.label
                })

    df = pd.DataFrame(mentions)
    return df

def fetch_zenodo_records(query, max_records = 200):
    """Fetch a few hundred records from Zenodo for canonical linking."""
    url = "https://zenodo.org/api/records"
    page = 1
    all_items = []
    # while len(all_items) < max_records:
    r = requests.get(url, params={"q": query, "size": 10, "page": page})
    hits = r.json().get("hits", {}).get("hits", [])

    for h in hits:
        md = h.get("metadata", {})
        title = md.get("title", "")
        doi = md.get("doi", "")
        resource_type = md.get("resource_type", {}).get("type", "")
        all_items.append({
            "canonical_id": str(h.get("id", "")),
            "name": title.strip(),
            "type": resource_type.lower(),
            "aliases": title.lower(),
            "homepage": h.get("links", {}).get("html", ""),
            "doi": doi,
            "source": "Zenodo"
        })

    # page += 1
    return all_items

# def keyword_matching(detected_entities, canonical_entities, *, TOPK=10, THRESH=0.7):
#     from rapidfuzz import process, fuzz
#     rows = []
#     for ent in tqdm(detected_entities, desc="Linking (keyword)"):
#         # top-1
#         best = process.extractOne(ent, canonical_entities, scorer=fuzz.token_sort_ratio)
#         # top-k (for Hits@k/MRR). Returns list of tuples (match, score, idx)
#         topk = process.extract(ent, canonical_entities, scorer=fuzz.token_sort_ratio, limit=TOPK)
#         if best is None:
#             pred_name, score = 'NIL', 0
#             topk_names = []
#         else:
#             pred_name, score, _ = best
#             if score < THRESH * 100:
#                 pred_name = 'NIL'
#             topk_names = [m[0] for m in topk] if topk else []

#         rows.append({
#             'entity_text': ent,
#             'pred_name': pred_name,
#             'score': score,
#             'candidates': topk_names,
#         })
#     df = pd.DataFrame(rows)
#     return df

def keyword_matching(nerdme_df: pd.DataFrame, canonical_entities: List[str], *, cfg: Config = Config()):
    from rapidfuzz import process, fuzz
    results = []
    for i, row in tqdm(nerdme_df.iterrows(), desc="Linking (keyword)"):
        ent = row['entity_text']
        # top-1
        best = process.extractOne(ent, canonical_entities, scorer=fuzz.token_sort_ratio)
        # top-k (for Hits@k/MRR). Returns list of tuples (match, score, idx)
        topk = process.extract(ent, canonical_entities, scorer=fuzz.token_sort_ratio, limit=cfg.topk)
        if best is None:
            pred_name, score = 'NIL', 0
            topk_names = []
        else:
            pred_name, score, _ = best
            if score < cfg.threshold * 100:
                pred_name = 'NIL'
            topk_names = [m[0] for m in topk] if topk else []

        results.append({
            'entity_text': ent,
            'pred_name': pred_name,
            'score': score,
            'candidates': topk_names,
            'gold_name': row['gold_name'],
            'entity_type': row['entity_type'],
        })
    df = pd.DataFrame(results)
    return df

def semantic_matching(nerdme_df: pd.DataFrame, canonical_entities: List[str], *, cfg: Config = Config()):
    import torch
    import sentence_transformers as st

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 2) Pre-encode canonical entities once (normalized embeddings)
    model = st.SentenceTransformer('all-MiniLM-L6-v2')
    model = model.to(device)
    detected_entities = nerdme_df['entity_text'].tolist()
    detected_emb = model.encode(detected_entities, convert_to_tensor=True)
    canonical_emb = model.encode(canonical_entities, convert_to_tensor=True)
    cosine_scores = st.util.cos_sim(detected_emb, canonical_emb)
    results = []
    for i, row in tqdm(nerdme_df.iterrows(), desc='Linking (semantic)'):
        ent = row['entity_text']
        scores = cosine_scores[i]
        topk_idx = scores.argsort().numpy()[::-1][:cfg.topk]
        best_idx = topk_idx[0]
        best_score = scores[best_idx].item()

        topk_names = [canonical_entities[j] for j in topk_idx]
        pred_name = canonical_entities[best_idx] if best_score >= cfg.threshold else 'NIL'
        results.append({
            "entity_text": ent,
            "pred_name": pred_name,
            "score": best_score,     # cosine in [-1,1]; typically 0.0–0.9 for good matches
            "candidates": topk_names,
            'gold_name': row['gold_name'],
            'entity_type': row['entity_type'],
        })

    df = pd.DataFrame(results)
    return df

def load_pwc_dataset():
    """
    Loads the Papers with Code dataset catalog from HuggingFace.
    Expects a dataset with a 'train' split and fields like 'name', 'full_name', 'homepage'.
    """
    pwc = load_dataset("pwc-archive/datasets")

    df = pwc["train"].to_pandas()

    # Build a canonical name field. You can tune this.
    # Here we prefer verified_name or name/full_name if available.
    def canonical_name(row):
        if isinstance(row.get("verified_name"), str) and row["verified_name"].strip():
            return row["verified_name"].strip()
        if isinstance(row.get("name"), str) and row["name"].strip():
            return row["name"].strip()
        if isinstance(row.get("full_name"), str) and row["full_name"].strip():
            return row["full_name"].strip()
        return None

    df["canonical_name"] = df.apply(canonical_name, axis=1)
    df = df.dropna(subset=["canonical_name"]).reset_index(drop=True)

    # Add a simple canonical id (row index) if you do not want to use URLs
    df["canonical_id"] = df.index.astype(int)

    return df
def load_zenodo_dataset(entities_to_query: List[str], target_label: str):
    all_items = []
    save_path = Path(f"results/entity-linking/zenodo/{target_label}.csv")
    ckpt_path = Path(f'results/entity-linking/zenodo/ckpt_{target_label.lower()}.txt')
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if ckpt_path.exists():
        saved_idx = int(ckpt_path.read_text())
        df = pd.read_csv(save_path)
        for i, row in df.iterrows():
            all_items.append(dict(row))
    else:
        saved_idx = 0

    last_saved_item_cnt = 0
    for idx, query in enumerate(entities_to_query):
        if idx <= saved_idx:
            continue

        items = fetch_zenodo_records(query, max_records=10)
        if len(items) == 0:
            logging.info(f"[{idx:5d}]-[{query}] has no corresponding {target_label} on Zenodo")

        all_items.extend(items)
        df = pd.DataFrame(all_items)
        df.to_csv(save_path, index=False)
        logging.info(f"[{idx:5d}]-[{query}] saved {len(all_items) - last_saved_item_cnt} Zenodo canonical entries to {save_path}")
        ckpt_path.write_text(str(idx))
        last_saved_item_cnt = len(all_items)
        time.sleep(0.3)

    df = pd.DataFrame(all_items)

    return df
def load_nerdme_mentions(file: Path):
    """
    Expects a CSV with at least:
        - entity_text  (the dataset mention in NERdME)
    Optionally:
        - gold_pwc_name or gold_pwc_id (for evaluation, if you have it)
    """
    df = pd.read_csv(path)
    # Basic cleaning
    df["entity_text"] = df["entity_text"].astype(str).str.strip()
    return df
def evaluate_linking(pred_df, gold_df, gold_col="gold_pwc_name"):
    """
    pred_df: output of keyword_matching or semantic_matching
    gold_df: NERdME mentions with a gold column specifying the correct PwC name
    gold_col: column in gold_df that stores the gold canonical name (string)
    """
    df = pred_df.merge(gold_df[["entity_text", gold_col]], on="entity_text", how="left")
    df[gold_col] = df[gold_col].fillna("NIL")

    # Classification metrics
    tp = ((df["pred_name"] == df[gold_col]) & (df[gold_col] != "NIL")).sum()
    fp = ((df["pred_name"] != df[gold_col]) & (df["pred_name"] != "NIL")).sum()
    fn = ((df["pred_name"] == "NIL") & (df[gold_col] != "NIL")).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Ranking metrics (Hits@k and MRR)
    def hits_at_k(row, k):
        gold = row[gold_col]
        if gold == "NIL":
            return 0.0
        return 1.0 if gold in row["candidates"][:k] else 0.0

    def reciprocal_rank(row):
        gold = row[gold_col]
        if gold == "NIL":
            return 0.0
        cand = row["candidates"]
        if gold in cand:
            r = cand.index(gold) + 1
            return 1.0 / r
        return 0.0

    hits1 = df.apply(lambda r: hits_at_k(r, 1), axis=1).mean()
    hits3 = df.apply(lambda r: hits_at_k(r, 3), axis=1).mean()
    mrr = df.apply(reciprocal_rank, axis=1).mean()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hits@1": hits1,
        "hits@3": hits3,
        "mrr": mrr,
    }


def main(args):
    # Prepare NERdME entity
    files = [x for x in Path("data/train").rglob("*.tsv")] + [x for x in Path("data/val").rglob("*.tsv")] + [x for x in Path("data/test_labeled/").rglob("*.tsv")]
    for target_label in LABELS:
        save_path = Path(f'results/entity-linking/nerdme/{target_label}.csv')
        if save_path.exists() and save_path.stat().st_size > 0:
            continue
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df = extract_entity_from_tsv(files, target_label=target_label)
        logging.info(f"✅ Extracted {len(df)} {target_label} entities → {save_path}")
        df.to_csv(save_path, index=False)

    gold_nerdme_entity_path = Path(f'results/entity-linking/nerdme/{args.target_label}_GOLD.csv')

    # Attach gold if available; otherwise default NIL
    if not gold_nerdme_entity_path.exists():
        nerdme_entity_path = Path(f'results/entity-linking/nerdme/{args.target_label}.csv')
        nerdme_df = pd.read_csv(nerdme_entity_path)
        nerdme_df['gold_name'] = nerdme_df['entity_text'].apply(lambda x: x.lower())
        nerdme_df.to_csv(gold_nerdme_entity_path, index=False)
    else:
        nerdme_df = pd.read_csv(gold_nerdme_entity_path)

    # detected_entities = nerdme_df['entity_text'].tolist()

    # entities_to_query = []
    # specials = ['/', '-', '|', '_', '+', '=', '>', '<']
    # for entity in detected_entities:
    #     entity = entity.lower()
    #     if '(' in entity:
    #         entity = entity.replace('(', '')
    #     if ')' in entity:
    #         entity = entity.replace(')', '')

    #     for sp in specials:
    #         if sp in entity:
    #             entity = entity.replace(sp, ' ')

    #     if entity in entities_to_query:
    #         continue

    #     entity = normalize_spaces(entity)
    #     entities_to_query.append(entity)
    # logging.info(f"NERDME unique entities: {len(entities_to_query)}")

    pwc_df = load_pwc_dataset()
    canonical_entities = pwc_df["canonical_name"].str.lower().tolist()
    # 3) Run keyword based linking
    kw_pred = keyword_matching(nerdme_df, canonical_entities)
    # kw_pred.to_csv("pwc_linking_keyword.csv", index=False)
    # # 4) Run semantic linking
    sem_pred = semantic_matching(nerdme_df, canonical_entities)
    # sem_pred.to_csv("pwc_linking_semantic.csv", index=False)
    import ipdb; ipdb.set_trace()
    sys.exit(0)

    # # datasets = list_datasets(limit=1000)
    # # rows = []
    # # for ds in datasets:
    # #     rows.append({
    # #         "canonical_id": ds.id.lower().replace("/", "_"),
    # #         "name": ds.id.split("/")[-1],
    # #         "type": "Dataset",
    # #         "aliases": ds.id,
    # #         "homepage": f"https://huggingface.co/datasets/{ds.id}",
    # #         "source": "HuggingFace Hub"
    # #     })

    # # df = pd.DataFrame(rows)
    # # output_csv = Path("./results/entity-linking/huggingface/huggingface.csv")
    # # output_csv.parent.mkdir(parents=True, exist_ok=True)
    # # df.to_csv(output_csv, index=False)
    # # logging.info(f"Saved {len(df)} canonical dataset entries to canonical_datasets_hf.csv")

    # # Example: canonical list from Papers with Code
    # # huggingface
    # # huggingface_entity_path = Path("./results/entity-linking/huggingface/huggingface.csv")
    # # huggingface_df = pd.read_csv(huggingface_entity_path)
    # # canonical_entities = huggingface_df['name'].tolist()


    # zenodo_df = pd.read_csv(zenodo_save_path)
    # # zenodo_df = zenodo_df[zenodo_df.type == 'dataset']
    # logging.info(f"[{args.target_label}] zenodo: {len(zenodo_df)}")
    # canonical_entities = zenodo_df['name'].tolist()
    # THRESH = 0.7  # your current threshold on RapidFuzz (0..100). Tune on a dev set.
    # TOPK = args.topk     # for optional Hits@k/MRR
    # BATCH = 256
    # # ----------------------------
    # # 2) Predict best match (top-1), keep top-k for ranking metrics
    # # ----------------------------
    # if args.method == 'keyword':
    #     pred_df = keyword_matching(detected_entities, canonical_entities, TOPK=args.topk, THRESH=THRESH)
    # elif args.method == 'semantic':
    #     pred_df = semantic_matching(detected_entities, canonical_entities, TOPK=args.topk, THRESH=THRESH)

    # # ----------------------------
    # # 4) Metrics
    # # ----------------------------
    # def safe_div(a, b):
    #     return a / b if b else 0.0

    # eval_df = nerdme_df[['entity_text']].merge(pred_df, on='entity_text', how='left')
    # import ipdb; ipdb.set_trace()
    # # eval_df['gold_is_nil'] = (eval_df['gold_name'].astype(str).str.upper() == 'NIL')
    # # eval_df['pred_is_nil'] = (eval_df['pred_name'].astype(str).str.upper() == 'NIL')
    # # eval_df['is_correct_link'] = (~eval_df['gold_is_nil']) & (~eval_df['pred_is_nil']) & (eval_df['gold_name'] == eval_df['pred_name'])

    # # (A) Disambiguation (only mentions with gold != NIL)
    # disamb = eval_df[~eval_df['gold_is_nil']]
    # gold_links = len(disamb)
    # pred_links_on_linkable = (~disamb['pred_is_nil']).sum()
    # correct_links = disamb['is_correct_link'].sum()

    # disamb_precision = safe_div(correct_links, pred_links_on_linkable)       # = correct_links / predicted_links
    # disamb_recall    = safe_div(correct_links, gold_links)                   # = correct_links / gold_links
    # disamb_f1        = safe_div(2*disamb_precision*disamb_recall, (disamb_precision + disamb_recall))

    # # (B) End-to-end with NIL classification
    # TP_nil = ((eval_df['gold_is_nil']) & (eval_df['pred_is_nil'])).sum()
    # FP_nil = ((eval_df['gold_is_nil']) & (~eval_df['pred_is_nil'])).sum()    # over-linking
    # FN_nil = ((~eval_df['gold_is_nil']) & (eval_df['pred_is_nil'])).sum()    # under-linking

    # nil_precision = safe_div(TP_nil, TP_nil + FP_nil)
    # nil_recall    = safe_div(TP_nil, TP_nil + FN_nil)
    # nil_f1        = safe_div(2*nil_precision*nil_recall, (nil_precision + nil_recall))

    # # Micro P/R/F1 over link decisions (treat NIL as "no link")
    # pred_links_all   = (~eval_df['pred_is_nil']).sum()
    # gold_links_all   = (~eval_df['gold_is_nil']).sum()
    # correct_links_all= eval_df['is_correct_link'].sum()

    # micro_p = safe_div(correct_links_all, pred_links_all)
    # micro_r = safe_div(correct_links_all, gold_links_all)
    # micro_f1= safe_div(2*micro_p*micro_r, (micro_p + micro_r))

    # # ----------------------------
    # # 5) Optional: Hits@k and MRR on linkable mentions using ranked candidates
    # # ----------------------------
    # def hits_at_k(row, k=1):
    #     if row['gold_is_nil']:
    #         return None
    #     cands = row['candidates'] if isinstance(row['candidates'], list) else []
    #     return int(row['gold_name'] in cands[:k])

    # def mrr_row(row):
    #     if row['gold_is_nil']:
    #         return None
    #     cands = row['candidates'] if isinstance(row['candidates'], list) else []
    #     if row['gold_name'] in cands:
    #         return 1.0 / (cands.index(row['gold_name']) + 1)
    #     return 0.0

    # # Compute only if you provided gold names (not NIL)
    # if (~eval_df['gold_is_nil']).any():
    #     hits1 = eval_df.apply(hits_at_k, axis=1, k=1).dropna().mean()
    #     hits3 = eval_df.apply(hits_at_k, axis=1, k=3).dropna().mean()
    #     hits5 = eval_df.apply(hits_at_k, axis=1, k=5).dropna().mean()
    #     mrr   = eval_df.apply(mrr_row, axis=1).dropna().mean()
    # else:
    #     hits1 = hits3 = hits5 = mrr = None

    # # ----------------------------
    # # 6) Print summary
    # # ----------------------------
    # logging.info("\n=== Entity Linking Evaluation ===")
    # logging.info(f"Threshold: {THRESH*100}")
    # logging.info(f"Total mentions: {len(eval_df)} | Linkable (gold!=NIL): {gold_links_all} | Predicted links: {pred_links_all}")
    # logging.info(f"[Disambiguation]  P={disamb_precision:.4f}  R={disamb_recall:.4f}  F1={disamb_f1:.4f}")
    # logging.info(f"[NIL Handling]    P={nil_precision:.4f}  R={nil_recall:.4f}  F1={nil_f1:.4f}")
    # logging.info(f"[Micro overall]   P={micro_p:.4f}  R={micro_r:.4f}  F1={micro_f1:.4f}")
    # if hits1 is not None:
    #     logging.info(f"[Ranking]         Hits@1={hits1:.4f}  Hits@3={hits3:.4f} Hits@5={hits5:.4f}  MRR={mrr:.4f}")

    # # # Optionally save the detailed table for error analysis
    # # eval_df.to_csv(f"./results/entity-linking/eval_predictions_{target_label}_{args.topk}.csv", index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False)
    parser.add_argument("--target_label", type=str, required=False, default='DATASET')
    parser.add_argument("--topk", type=int, required=False, default=5)
    parser.add_argument("--method", type=str, choices=['keyword', 'semantic'])
    args = parser.parse_args()

    main(args)
