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

from predictor import BasePredictor, LABELS
from webanno_tsv import webanno_tsv_read_file, Document, Annotation, Token
import utils
import cleaner
from difflib import SequenceMatcher

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
            result_pred.append(colored('+'*(j2-j2), 'yellow'))
            adjusted_ops.append((tag, i1, i2, j1, j2))
        elif tag == "insert":
            result_ref.append(colored('-'*(i2-i1), 'red'))
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


# def char_diff_converter(ref, pred, ops):
#     result_ref = []
#     result_pred = []

#     for tag, i1, i2, j1, j2 in ops:
#         if tag == "equal":
#             result_pred.append(pred[i1:i2])
#             result_ref.append(ref[j1:j2])
#         else:
#             if tag == "delete":
#                 # result_pred.append(pred[i1:i2])
#                 result_pred.append(ref[j1:j2])
# #                result_ref.append(ref[j1:j2])
#             elif tag == "insert":
#                 result_pred.append("@"*(j2-j1))
# #                result_ref.append(ref[j1:j2])
#             elif tag == 'replace':
#                # (i1, i2) -> (j1, j2)
#                if (i2 - i1) > (j2 - j1):
#                    result_pred.append('+'*(i2-i1))
#                else:
#                    result_pred.append('-'*(i2-i1))
#                 # result_pred.append(ref[j1:j2])
# #                result_ref.append(ref[j1:j2])
# #    ref, pred = ''.join(result_ref), ''.join(result_pred)
#     pred = ''.join(result_pred)
#     # if ref != pred:
#     #     import ipdb; ipdb.set_trace()
#     return ref, pred, ops

import re
from difflib import SequenceMatcher

import re

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

# Join tags into pattern
tag_pattern = '|'.join(LABELS)


tag_pattern = '|'.join(LABELS)
TAG_RE = re.compile(rf"<(?P<tag>{tag_pattern})>(?P<content>.*?)</(?P=tag)>")

def extract_tagged_spans(text):
    spans = []

    def replace(match):
        tag = match.group("tag")
        content = match.group("content")
        spans.append({'tag': tag, 'content': content})
        return content  # Replace tagged span with plain content

    cleaned = TAG_RE.sub(replace, text)
    return cleaned, spans

# def extract_tagged_spans(text):
#     tag_pattern = r"<(?P<tag>\w+)>(?P<content>.*?)</\1>"
#     spans = []

#     def _replacer(match):
#         spans.append({
#             "tag": match.group("tag"),
#             "content": match.group("content")
#         })
#         return match.group("content")

#     stripped_text = re.sub(tag_pattern, _replacer, text)
#     return stripped_text, spans


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

def transfer_tags(pred_raw_text, ref_text):
    stripped_pred, spans = extract_tagged_spans(pred_raw_text)
    tagged_ref = reinsert_tags_with_fuzzy(ref_text, spans)
    return tagged_ref



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class OpenAIPredictor(BasePredictor):
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
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
        path = f'results/{self.model_name}/prompt-{self.prompt_id}/zzz_{self.file_name}' # NOTE: prefix zzz for directory sorting, non-sense
        os.makedirs(path, exist_ok=True)
        sid = hashlib.sha256(sentence.text.encode()).hexdigest()[:8]
        if not os.path.isfile(f'{path}/{sid}.txt'):
            self.do_prediction(sentence, f'{path}/{sid}.txt')

        with open(f'{path}/{sid}.txt', 'r') as fd:
            predicted_text = fd.read()

        # label_to_text_list, pure_pred_text = self.post_process(predicted_text)
        # for label, text_list in label_to_text_list.items():
        #     for text in text_list:
        #         text['sentence_idx'] = sentence.idx
        #         self.label_to_text_list[label].append(text)

        # import ipdb; ipdb.set_trace()
        ref_text = sentence.text
        cleaned_text = cleaner.Cleaner(predicted_text).clean()
        tagged_ref = transfer_tags(cleaned_text, ref_text)
        label_to_text_list, pure_pred_text = self.post_process(tagged_ref)
        if pure_pred_text != ref_text:
            logging.info(f"ref text          : {colored(ref_text, 'green')}")
            logging.info(f"retagged pred text: {colored(tagged_ref, 'red')}")
            logging.info(f"pure pred text : {colored(pure_pred_text, 'cyan')}")

        else:
            logging.info('GOOD')
        # logging.info(f"ref text          : {colored(ref_text, 'green')}")
        # logging.info(f"retagged pred text: {colored(tagged_ref, 'red')}")
        # logging.info(f"cleaned pred text : {colored(cleaned_text, 'cyan')}")
        # logging.info('--------------------------------------------------')
        # import ipdb; ipdb.set_trace()
        # if ref_text != pure_pred_text:
        #     logging.warning("Text not match: ")
        #     logging.info(f"ref text     : {colored(ref_text, 'green')}")
        #     logging.info(f"pred text    : {colored(pure_pred_text, 'red')}")
        #     logging.info(f"pred raw text: {predicted_text}")
        #     # logging.info(f"ref  text: {ref_text}")
        #     # logging.info(f"pred text: {pure_pred_text}")
        #     # logging.info(f"pred text: {predicted_text}")
        #     # x, y = char_diff(ref_text, pure_pred_text)
        #     logging.info("-------------------------------------------------------------------")
        #     x, y, ops = char_diff(ref_text, pure_pred_text)
        #     # logging.info(f"ref text     : {y}")
        #     # logging.info(f"pred text    : {x}")
        #     # logging.info("-------------------------------------------------------------------")
        #     # x, y, ops = char_diff_converter(ref_text, pure_pred_text)
        #     # x, y, ops = char_diff_converter(ref_text, pure_pred_text, ops)
        #     # if ref_text != y:
        #     #     logging.info("BUG: wrong x and ref_text")
        #     # import ipdb; ipdb.set_trace()
        #     logging.info(f"ref text     : {x}")
        #     logging.info(f"pred text    : {y}")
        #     logging.info(f'ops:           {colored(ops, "cyan")}')
        #     logging.info(f'label_to_text_list:  {label_to_text_list}')
        #     logging.info("############################################")

        #     for label, text_list in label_to_text_list.items():
        #         # logging.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        #         for text in text_list:
        #             start, end = text['start'], text['end']
        #             # logging.info(f"before alignemnt, start {text['start']}, end: {text['end']}")
        #             for tag, i1, i2, j1, j2 in ops:
        #                 if tag == 'insert':
        #                     if start >= j2:
        #                         # (j1, j2) start, end ...
        #                         text['start'] -= (j2 - j1)
        #                         text['end'] -= (j2 - j1)
        #                     elif end < j1:
        #                         # start, end (j1, j2)
        #                         pass
        #                     elif start < j1 and j2 < end:
        #                         # start < j1, j2 < end
        #                         text['end'] -= (j2 - j1)
        #                     elif j1 < end and j2 < end:
        #                         # start < j1 < end < j2
        #                         # start < j1, j2 < end
        #                         # consider it as false, we don't want to handle it
        #                         pass
        #                 elif tag == 'delete':
        #                     if start >= j2:
        #                         # (i1, i2) start, end
        #                         text['start'] += (i2 - i1)
        #                         text['end'] += (i2 - i1)
        #                     elif end < j1:
        #                         # start, end, (i1, i2)
        #                         pass
        #                     elif start < j1 and j2 < end:
        #                         # start, i1, i2, end
        #                         text['end'] += (i2 - i1)
        #                     elif i1 < end and end < i2:
        #                         # consider it as false, we don't want to handle it
        #                         pass
        #                 elif tag == 'replace':
        #                     if start >= j2:
        #                         # (i1, xxxxx -> (j1, j2) (start, end)
        #                         text['start'] += (i2 - i1) - (j2 - j1)
        #                         text['end'] += (i2 - i1) - (j2 - j1)

        #                     elif start >= j1 and end <= j2:
        #                         # (i1, start, end, i2) -> (j1, start, end, j2)
        #                         # text['start'] += start-i1 - (start - j1)
        #                         # import ipdb; ipdb.set_trace()
        #                         text['start'] = i1
        #                         text['end'] = i2
        #                         # text['text'] = ref_text[i1:i2]
        #                         start, end = i1, i2
        #                 # logging.info(f"after alignemnt {tag}, start {text['start']}, end: {text['end']}")

        #             start = text['start']
        #             end = text['end']
        #             if text['text'].lower() != ref_text[start:end].lower():
        #                 logging.error(f"BUG: \n > {ref_text[start:end]}\n > {text['text']}")
        #                 import ipdb; ipdb.set_trace()
        #             else:
        #                 logging.info(f'{label}: {ref_text[start:end]}')


        #     # logging.info(f'after adjust label_to_text_list:  {label_to_text_list}')

        #     # for label, text_list in label_to_text_list.items():
        #     #     for text in text_list:
        #     #         start = text['start']
        #     #         end = text['end']
        #     #         logging.info(f'{label}: {y[start:end]}')

        #     #         if text['text'] != y[start:end]:
        #     #             logging.error("BUG")
        #     #             import ipdb; ipdb.set_trace()

        #     self.mismatch_sentences += 1
        # else:
        #     pass
        #     # logging.warning("Text match: ")
        #     # logging.info(f"ref text     : {colored(ref_text, 'green')}")
        #     # logging.info(f"pred text    : {colored(pure_pred_text, 'cyan')}")
        #     # logging.info(f"pred raw text: {predicted_text}")
        #     # found = False
        #     # for label in LABELS:
        #     #     if f'<{label}>' in predicted_text:
        #     #         found = True
        #     #         break

        #     # if found is True:
        #     #     logging.info("predicted text contains labels")
        #     # else:
        #     #     logging.info("predicted text *NOT* contains labels")
            # logging.info("================================================================================")

        # self.total_sentences += 1
        # NOTE: sanity checking
        # for label, text_list in label_to_text_list.items():
        #     for text in text_list:
        #         if text['text'] != sentence.text[text['start']:text['end']]:
        #             prompt = self.prompt_template.replace('{input_text}', sentence.text)
        #             # logging.warning(f"BUG? The predicted text is not exact the same as the original text. \n\nPrompt: {prompt}\nOriginal: {colored(sentence.text, 'green')}\nGenerated: {colored(text['text'], 'red')}\n--------------------------------------------------------------------------------")
        #             logging.warning(f"BUG? The predicted text is not exact the same as the original text. \n\nOriginal: {colored(sentence.text, 'green')}\nGenerated: {colored(text['text'], 'red')}\n--------------------------------------------------------------------------------")

        span_tokens_to_label_list = []
        for label, text_list in label_to_text_list.items():
            for text in text_list:
                span_tokens_to_label_list.append({
                    'span_tokens': utils.make_span_tokens(tokens, text['start'], text['end']),
                    'label': label
                })
        return span_tokens_to_label_list

    def post_process(self, predicted_text):
        cleaned_text = cleaner.Cleaner(predicted_text).clean()
        label_to_text_list, pure_text = self.extract_annotation_labels_if_possible(cleaned_text)

        return label_to_text_list, pure_text

    def extract_annotation_labels_if_possible(self, predicted_text):
        label_to_text_list = defaultdict(list)
        matched_labels = {}

        for label in LABELS:
            regex = f'<{label}>(.*?)</{label}>'
            matches = re.finditer(regex, predicted_text, flags=re.IGNORECASE | re.DOTALL)
            for m in matches:
                matched_labels[m.start(1)] = label

        acc_adjusted_pos = 0

        for pos in sorted(matched_labels):
            label = matched_labels[pos]
            regex = f'<{label}>(.*?)</{label}>'
            matches = re.finditer(regex, predicted_text, flags=re.IGNORECASE | re.DOTALL)
            matches = [x for x in matches]

            for m in matches:
                if m.start(1) != pos:
                    continue

                adjusted_pos = len(label) + 2
                start = m.start(1) - adjusted_pos - acc_adjusted_pos
                end = m.end(1) - adjusted_pos - acc_adjusted_pos
                if start < 0:
                    logging.error("BUG!!! please fix it")

                label_to_text_list[label].append({
                    'text': m.group(1),
                    'start': start,
                    'end': end
                })

                acc_adjusted_pos += adjusted_pos * 2 + 1
                # print(f"acc_adjusted_pos: {acc_adjusted_pos}, label: {label}, pos: {pos}")

        for label in LABELS:
            predicted_text = re.sub(f'<{label}>', '', predicted_text)
            predicted_text = re.sub(f'</{label}>', '', predicted_text)

        return label_to_text_list, predicted_text

    def do_prediction(self, sentence, sid_path):
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        try:
            print(f"Process-{os.getpid()} processing {colored(sentence.text, 'red')} ...")
            prompt = self.prompt_template.replace('{input_text}', sentence.text)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ]
            )
            print(f"Process-{os.getpid()} predict {colored(sentence.text, 'cyan')} successfully")
            result = response.choices[0].message.content
            with open(sid_path, 'w') as file:
                file.write(result)
        except Exception as ex:
            logging.error(f'[do_prediction] got exception: {ex}')

    def set_file_name(self, file_name):
        self.file_name = file_name

if __name__ == "__main__":
    # mp.set_start_method('fork')
    phase = 'test_labeled'
    base_path = f'./data/{phase}'
    file_names = [fp for fp in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, fp)) and fp.endswith('.tsv')]
    model_name = 'deepseek-chat'
    output_folder = f'./results/{model_name}/{phase}'
    os.makedirs(output_folder, exist_ok=True)
    # DeepSeek Chat
    predictor = OpenAIPredictor(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url='https://api.deepseek.com',
        model_name=model_name
    )

    for file_name in file_names:
        # if 'ARM-software_keyword-transformer_master_README.md.tsv' not in file_name:
        #     continue
        predictor.set_file_name(file_name)
        file_path = os.path.join(base_path, file_name)
        ref_doc = webanno_tsv_read_file(file_path)
        predicted_doc = predictor(ref_doc)

        # print(ref_doc.text)
        # for anno in ref_doc.annotations:
        #     print(f'{anno.label} -> {anno.text}')

        # print('----------------------------------------')
        # # for anno in predicted_doc.annotations:
        # #     print(f'{anno.label} -> {anno.text}')
        # for label, text_list in predictor.label_to_text_list.items():
        #     for text in text_list:
        #         print(f'{label} -> {text["text"]}')
        # Verify
        if ref_doc.text != predicted_doc.text:
            logging.debug('content changed')
        if len(ref_doc.sentences) == len(predicted_doc.sentences):
            logging.warning('sentences changed')
        if len(ref_doc.tokens) == len(predicted_doc.tokens):
            logging.debug('tokens changed')
        for s1, s2 in zip(ref_doc.sentences, predicted_doc.sentences):
            if s1 == s2:
                logging.debug(f'sentence changed, \n{s1}\n{s2}')

        for t1, t2 in zip(ref_doc.tokens, predicted_doc.tokens):
            if t1 == t2:
                logging.debug(f'token changed: \n{t1}\n{t2}')

        logging.info(f"Predicted {len(predicted_doc.annotations)} annotations, mismatch_sentences/total sentences: {predictor.mismatch_sentences}/{predictor.total_sentences}")
        prediction_path = os.path.join(output_folder, file_name)
        with open(prediction_path, 'w') as fd:
            fd.write(predicted_doc.tsv())
