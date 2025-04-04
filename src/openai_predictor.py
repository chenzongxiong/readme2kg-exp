import sys
import os
import random
from openai import OpenAI
import hashlib
from predictor import BasePredictor, LABELS
from webanno_tsv import webanno_tsv_read_file, Document, Annotation, Token
import utils


class OpenAIPredictor(BasePredictor):
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.prompt_id = 0
        prompt_template_path = f'config/{model_name}-prompt-{self.prompt_id}.txt'
        if os.path.isfile(prompt_template_path):
            with open(prompt_template_path, 'r') as fd:
                self.prompt_template = fd.read()
        else:
            self.prompt_template = ''

    def predict(self, sentence, tokens):
        path = f'results/{self.model_name}/prompt-{self.prompt_id}/{self.file_name}'
        os.makedirs(path, exist_ok=True)
        sid = hashlib.sha256(sentence.text.encode()).hexdigest()
        if not os.path.isfile(f'{path}/{sid}.txt'):
            self.do_prediction(sentence, tokens, f'{path}/{sid}.txt')

        with open(f'{path}/{sid}.txt', 'r') as fd:
            predicted_text = fd.read()

        return self.post_process(predicted_text, tokens)

    def post_process(self, predicted_text, tokens):
        # TODO: return the span tokens, and predicted label
        # start_char = random.randint(tokens[0].start, tokens[-1].end - 1)
        # end_char = random.randint(start_char + 1, tokens[-1].end)
        # # NOTE: Since our NER task is character-level, we need handle the tokens at the edge of span carefully.
        # span_tokens = utils.make_span_tokens(tokens, start_char, end_char)
        # if span_tokens is None:
        #     return None, None
        # # Random select a label from LABELS
        # label = random.choice(LABELS)
        # return span_tokens, label
        import sys
        sys.exit(0)

    def do_prediction(self, sentence, tokens, sid_path):
        try:
            import ipdb; ipdb.set_trace()
            prompt = self.prompt_template.replace('{input_text}', sentence.text)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ]
            )
            result = response.choices[0].message.content
            with open(sid_path, 'w') as file:
                file.write(result)

            # sys.exit(1)
        except Exception as ex:
            # raise f"Error: {e}"
            raise ex

    def set_file_name(self, file_name):
        self.file_name = file_name

if __name__ == "__main__":
    base_path = './data/train'
    file_names = [fp for fp in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, fp)) and fp.endswith('.tsv')]
    model_name = 'deepseek-chat'
    output_folder = f'./results/{model_name}'
    os.makedirs(output_folder, exist_ok=True)
    # DeepSeek Chat
    predictor = OpenAIPredictor(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url='https://api.deepseek.com',
        model_name=model_name
    )

    for file_name in file_names:
        predictor.set_file_name(file_name)
        file_path = os.path.join(base_path, file_name)
        ref_doc = webanno_tsv_read_file(file_path)
        predicted_doc = predictor(ref_doc)
        break
        # Verify
        # assert ref_doc.text == predicted_doc.text, 'content changed'
        # assert len(ref_doc.sentences) == len(predicted_doc.sentences), 'sentences changed'
        # assert len(ref_doc.tokens) == len(predicted_doc.tokens), 'tokens changed'
        # for s1, s2 in zip(ref_doc.sentences, predicted_doc.sentences):
        #     assert s1 == s2, f'sentence changed, \n{s1}\n{s2}'

        # for t1, t2 in zip(ref_doc.tokens, predicted_doc.tokens):
        #     assert t1 == t2, f'token changed: \n{t1}\n{t2}'

        # print(f"Predicted {len(predicted_doc.annotations)} annotations")
        # prediction_path = os.path.join(output_folder, file_name)
        # with open(prediction_path, 'w') as fd:
        #     fd.write(predicted_doc.tsv())
