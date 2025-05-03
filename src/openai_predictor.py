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

logging.basicConfig(
    filename='logs/deepseek-chat.log',
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
            self.do_prediction(sentence, tokens, f'{path}/{sid}.txt')

        with open(f'{path}/{sid}.txt', 'r') as fd:
            predicted_text = fd.read()

        label_to_text_list = self.post_process(predicted_text, tokens)
        # NOTE: sanity checking
        for label, text_list in label_to_text_list.items():
            for text in text_list:
                if text['text'] != sentence.text[text['start']:text['end']]:
                    prompt = self.prompt_template.replace('{input_text}', sentence.text)
                    logging.warning(f"BUG? The predicted text is not exact the same as the original text. \n\nPrompt: {prompt}\nOriginal: {colored(sentence.text, 'green')}\nGenerated: {colored(text['text'], 'red')}\n--------------------------------------------------------------------------------")

        span_tokens_to_label_list = []
        for label, text_list in label_to_text_list.items():
            for text in text_list:
                span_tokens_to_label_list.append({
                    'span_tokens': utils.make_span_tokens(tokens, text['start'], text['end']),
                    'label': label
                })
        return span_tokens_to_label_list

    def post_process(self, predicted_text, tokens):
        # TODO: Debug the span of generated files
        cleaned_text = cleaner.Cleaner(predicted_text).clean()
        label_to_text_list = self.extract_annotation_labels_if_possible(cleaned_text)
        return label_to_text_list

    def extract_annotation_labels_if_possible(self, predicted_text):
        label_to_text_list = defaultdict(list)
        acc_adjusted_pos = 0

        matched_labels = {}

        for label in LABELS:
            regex = f'<{label}>(.*?)</{label}>'
            matches = re.finditer(regex, predicted_text, flags=re.IGNORECASE | re.DOTALL)
            for m in matches:
                matched_labels[m.start(1)] = label

        for pos in sorted(matched_labels):
            label = matched_labels[pos]
            regex = f'<{label}>(.*?)</{label}>'
            matches = re.finditer(regex, predicted_text, flags=re.IGNORECASE | re.DOTALL)
            for m in matches:
                adjusted_pos = len(label) + 2
                label_to_text_list[label].append({
                    'text': m.group(1),
                    'start': m.start(1) - adjusted_pos - acc_adjusted_pos,
                    'end': m.end(1) - adjusted_pos - acc_adjusted_pos,
                })
                acc_adjusted_pos += adjusted_pos * 2 + 1

        return label_to_text_list

    def do_prediction(self, sentence, tokens, sid_path):
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
    phase = 'test_unlabeled'
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
        # Verify
        if ref_doc.text != predicted_doc.text:
            logging.warning('content changed')
        if len(ref_doc.sentences) == len(predicted_doc.sentences):
            logging.warning('sentences changed')
        if len(ref_doc.tokens) == len(predicted_doc.tokens):
            logging.warning('tokens changed')
        for s1, s2 in zip(ref_doc.sentences, predicted_doc.sentences):
            if s1 == s2:
                logging.warning(f'sentence changed, \n{s1}\n{s2}')

        for t1, t2 in zip(ref_doc.tokens, predicted_doc.tokens):
            if t1 == t2:
                logging.warning(f'token changed: \n{t1}\n{t2}')

        logging.warning(f"Predicted {len(predicted_doc.annotations)} annotations")
        prediction_path = os.path.join(output_folder, file_name)
        with open(prediction_path, 'w') as fd:
            fd.write(predicted_doc.tsv())
