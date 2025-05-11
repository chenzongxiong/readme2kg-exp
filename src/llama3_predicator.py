import re
import sys
import os
import random
from collections import defaultdict
from termcolor import colored
from openai import OpenAI
from functools import partial, reduce
import operator as op
import hashlib
import multiprocessing as mp
import logging
from pathlib import Path
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import utils
import cleaner
from webanno_tsv import webanno_tsv_read_file, Document, Annotation, Token
from predictor import BasePredictor, LABELS


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# torch.cuda.is_available()
# device = "cuda" if torch.cuda.is_available() else "cpu"


def char_diff(ref: str, pred: str):
    matcher = SequenceMatcher(None, ref, pred)
    result_ref = []
    result_pred = []
    adjusted_ops = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        if tag == "equal":
            result_ref.append(ref[i1:i2])
            result_pred.append(pred[j1:j2])
            adjusted_ops.append((tag, i1, i2, j1, j2))
        elif tag == "replace":
            result_ref.append(colored(ref[i1:i2], 'red'))
            result_pred.append(colored(pred[j1:j2], 'cyan'))
            # if i1 != i2:
            #     adjusted_ops.append(('delete', i1, i2, j1, j1))
            # if j1 != j2:
            #     adjusted_ops.append(('insert', i1, i1, j1, j2))
            adjusted_ops.append((tag, i1, i2, j1, j2))
        elif tag == "delete":
            result_ref.append(colored(ref[i2:i1], 'red'))
            result_pred.append(colored('+'*(i2-i2), 'yellow'))
            adjusted_ops.append((tag, i1, i2, j1, j2))
        elif tag == "insert":
            result_ref.append(colored('-'*(j2-j1), 'red'))
            result_pred.append(colored(pred[j2:j1], 'green'))
            adjusted_ops.append((tag, i1, i2, j1, j2))


    ops = []
    for op in adjusted_ops:
        if op[0] == 'equal':
            continue
        if op[0] == 'replace':
            ops.insert(0, op)
        else:
            ops.append(op)
    return ''.join(result_ref), ''.join(result_pred), ops


def fuzzy_find_span(content, ref_text, used_ranges):
    matcher = SequenceMatcher(None, ref_text, content)
    best_match = None
    best_ratio = 0.0

    for block in matcher.get_matching_blocks():
        start = block.a
        end = start + block.size
        if block.size == 0:
            continue
        # Ignore overlaps
        if any(start < r[1] and end > r[0] for r in used_ranges):
            continue
        ratio = SequenceMatcher(None, ref_text[start:end], content).ratio()
        if ratio > best_ratio and content in ref_text[start:end]:
            best_match = (start, end)
            best_ratio = ratio

    return best_match


def reinsert_tags_with_fuzzy(ref_text, spans):
    tagged_text = ref_text
    used_ranges = []
    for span in sorted(spans, key=lambda s: -len(s["content"])):  # Longest first
        tag = span["tag"]
        content = span["content"]
        match = fuzzy_find_span(content, tagged_text, used_ranges)
        if match:
            start, end = match
            before = tagged_text[:start]
            tagged = f"<{tag}>{tagged_text[start:end]}</{tag}>"
            after = tagged_text[end:]
            tagged_text = before + tagged + after
            used_ranges.append((start, end + len(tag)*2 + 5))  # crude tag length estimate

    return tagged_text


def extract_nested_tags(tagged_text):
    """Extract (label, content) from nested tag structure."""
    labels = [x.lower() for x in LABELS]
    soup = BeautifulSoup(f"<root>{tagged_text}</root>", "html.parser")
    spans = []

    def recurse(tag):
        for child in tag.children:
            if getattr(child, 'name', None) in labels:
                content = ''.join(child.strings).strip()
                spans.append({'tag': child.name.upper(), 'content': content})
                recurse(child)  # Handle nested
            elif hasattr(child, 'children'):
                recurse(child)

    recurse(soup.root)
    return spans


def transfer_tags(pred_text, ref_text):
    spans = extract_nested_tags(pred_text)
    tagged_ref = reinsert_tags_with_fuzzy(ref_text, spans)
    return spans, tagged_ref


def find_spans_in_target(spans, target):
    """
    Match each (label, content) from spans into `target`,
    considering repeated content and preserving order.
    """
    used = [False] * len(target)
    result = []
    last_match_end = 0  # Avoid overlapping or backward matching

    for span in spans:
        label = span['tag']
        content = span['content']

        pattern = re.escape(content.strip())
        matches = list(re.finditer(pattern, target))

        # Try to match the next unused appearance after last_match_end
        found = False
        for m in matches:
            span_range = range(m.start(), m.end())
            if not any(used[i] for i in span_range) and m.start() >= last_match_end:
                for i in span_range:
                    used[i] = True
                result.append((label, content, m.start(), m.end()))
                last_match_end = m.end()
                found = True
                break

        if not found:
            # fallback: match the first unused occurrence anywhere
            for m in matches:
                span_range = range(m.start(), m.end())
                if not any(used[i] for i in span_range):
                    for i in span_range:
                        used[i] = True
                    result.append((label, content, m.start(), m.end()))
                    last_match_end = m.end()
                    break

    return result


class Llama3Predictor(BasePredictor):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model_name = model_id.split('/')[-1]
        self.prompt_id = 0
        prompt_template_path = f'config/{model_name}-prompt-{self.prompt_id}.txt'
        if os.path.isfile(prompt_template_path):
            with open(prompt_template_path, 'r') as fd:
                self.prompt_template = fd.read()
        else:
            self.prompt_template = ''

        self.parallel = False
        self.mismatch_sentences = 0
        self.total_sentences = 0

        self.label_to_text_list = defaultdict(list)
        self.text_with_tags_to_pure_text_list = []
        self.pure_text_to_text_with_tags_list = []

        self.tokenizer = None
        self.model = None

    def __call__(self, doc: Document):
        if getattr(self, 'parallel', False):
            return self.__call__parallel(doc)

        return self.__call__serial(doc)

    def __call__serial(self, doc: Document):
        annotations = []
        for sent in doc.sentences:
            tokens = doc.sentence_tokens(sent)
            span_tokens_to_label_list = self.predict(sentence=sent, tokens=tokens)

            # create the annotation instances
            for span_tokens_to_label in span_tokens_to_label_list:
                span_tokens = span_tokens_to_label['span_tokens']
                label = span_tokens_to_label['label']
                if span_tokens is None:
                    continue

                annotation = utils.make_annotation(tokens=span_tokens, label=label)
                annotations.append(annotation)

        result = utils.replace_webanno_annotations(doc, annotations=annotations)
        return result

    def __call__parallel(self, doc: Document):
        args_list = []
        for sent in doc.sentences:
            tokens = doc.sentence_tokens(sent)
            args_list.append((sent, tokens))

        with mp.Pool(64) as pool:
            span_tokens_to_label_list = pool.map(self._predict_wrapper, args_list)
            span_tokens_to_label_list = reduce(op.concat, span_tokens_to_label_list)

        annotations = []
        for span_tokens_to_label in span_tokens_to_label_list:
            span_tokens = span_tokens_to_label['span_tokens']
            label = span_tokens_to_label['label']
            if span_tokens is None:
                continue

            annotation = utils.make_annotation(tokens=span_tokens, label=label)
            annotations.append(annotation)

        result = utils.replace_webanno_annotations(doc, annotations=annotations)
        return result

    def _predict_wrapper(self, args):
        sentence, tokens = args
        return self.predict(sentence, tokens)

    def predict(self, sentence, tokens):
        # NOTE: prefix zzz for directory sorting, non-sense
        path = f'results/{self.model_name}/prompt-{self.prompt_id}/zzz_{self.file_name}'
        os.makedirs(path, exist_ok=True)
        sid = hashlib.sha256(sentence.text.encode()).hexdigest()[:8]
        if not os.path.isfile(f'{path}/{sid}.txt'):
            self.do_prediction(sentence, f'{path}/{sid}.txt')

        with open(f'{path}/{sid}.txt', 'r') as fd:
            predicted_text = fd.read()

        cleaned_text = cleaner.Cleaner(predicted_text).clean()
        ref_text = sentence.text
        print('ref_text:   ', colored(ref_text, 'red'))
        print('clean text: ', colored(cleaned_text, 'cyan'))

        spans, tagged_ref = transfer_tags(cleaned_text, ref_text)
        tagged_spans = extract_nested_tags(tagged_ref)

        # if len(spans) != len(tagged_spans):
        #     import ipdb; ipdb.set_trace()

        matches = find_spans_in_target(tagged_spans, ref_text)
        label_to_text_list = defaultdict(list)
        for label, content, start, end in matches:
            label_to_text_list[label.upper()].append({'text': content, 'start': start, 'end': end})

        # NOTE: Double check
        # Ensure the text extracted from predicted text is exact the same as reference text
        for label in label_to_text_list:
            text_list = label_to_text_list[label]
            for text in text_list:
                x = ref_text[text['start']:text['end']]
                y = text['text']
                if x != y:
                    logging.info(f"bug\n> {x}\n> {y}")
                    import ipdb; ipdb.set_trace()

        span_tokens_to_label_list = []
        for label, text_list in label_to_text_list.items():
            for text in text_list:

                span_tokens_to_label_list.append({
                    'span_tokens': utils.make_span_tokens(tokens, text['start'], text['end'])[0],
                    'span_tokens_debug': utils.make_span_tokens(tokens, text['start'], text['end'])[1],
                    'label': label
                })
                span_tokens = span_tokens_to_label_list[-1]['span_tokens']
                span_tokens_debug = span_tokens_to_label_list[-1]['span_tokens_debug']
                try:
                    annotation = utils.make_annotation(tokens=span_tokens, label=label)
                    if annotation.text != text['text']:
                        import ipdb; ipdb.set_trace()
                except Exception as ex:
                    import ipdb; ipdb.set_trace()

        return span_tokens_to_label_list

    def do_prediction(self, sentence, sid_path):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, force_download=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                force_download=True
            )
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        print(f"Process-{os.getpid()} processing {colored(sentence.text, 'red')} ...")
        prompt = self.prompt_template.replace('{input_text}', sentence.text)

        messages = [
            {"role": "system", "content": "You are a helpful NER annotator."},
            {"role": "user", "content": prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = outputs[0][input_ids.shape[-1]:]
        result = self.tokenizer.decode(response, skip_special_tokens=True)

        with open(sid_path, 'w') as file:
            file.write(result)

    def set_file_name(self, file_name):
        self.file_name = file_name


def double_check(ref_doc, predicted_doc, file_name):
    if ref_doc.text != predicted_doc.text:
        logging.warning(f'{file_name} content changed')
    if len(ref_doc.sentences) != len(predicted_doc.sentences):
        logging.warning(f'{file_name} sentences changed, {len(ref_doc.sentences)}/{len(predicted_doc.sentences)}')
    if len(ref_doc.tokens) != len(predicted_doc.tokens):
        logging.debug(f'{file_name} tokens changed')
    for s1, s2 in zip(ref_doc.sentences, predicted_doc.sentences):
        if s1 != s2:
            logging.warning(f'{file_name} sentence changed, \n{s1}\n{s2}')

    for t1, t2 in zip(ref_doc.tokens, predicted_doc.tokens):
        if t1 != t2:
            logging.warning(f'token changed: \n{t1}\n{t2}')


if __name__ == "__main__":
    phase = 'test_unlabeled'
    base_path = Path(f'data/{phase}')
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    model_name = model_id.split('/')[-1]
    file_paths = [x for x in base_path.rglob('*.tsv')]
    output_folder = Path(f'results/{model_name}/{phase}')
    os.makedirs(output_folder, exist_ok=True)

    predictor = Llama3Predictor(
        model_id=model_id
    )

    for file_path in file_paths:
        print(f'file_name: {file_path.name}')

        # if 'prasunroy_air-writing_master_README.md.tsv' not in file_path.name:
        #     continue
        predictor.set_file_name(file_path.name)
        ref_doc = webanno_tsv_read_file(file_path)
        pred_doc = predictor(ref_doc)
        double_check(ref_doc, pred_doc, file_path.name)
        prediction_path = output_folder / file_path.name
        with open(prediction_path, 'w') as fd:
            fd.write(pred_doc.tsv())
