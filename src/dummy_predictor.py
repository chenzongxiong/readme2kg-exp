import random
from collections import defaultdict
from webanno_tsv import webanno_tsv_read_file, Document, Annotation, Token
import utils
from predictor import LABELS, BasePredictor


class DummpyPredictor(BasePredictor):
    def predict(self, sentence, tokens):
        span_tokens_to_label_list = []
        annotations = []
        for label in LABELS:
            span_tokens_to_label = {
                'span_tokens': tokens,
                'label': label
            }
            span_tokens_to_label_list.append(span_tokens_to_label)
        return span_tokens_to_label_list


if __name__ == "__main__":
    import os
    base_path = './data/test_unlabeled'
    file_names = [fp for fp in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, fp)) and fp.endswith('.tsv')]
    output_folder = './results/dummy'
    os.makedirs(output_folder, exist_ok=True)

    predictor = DummpyPredictor()
    for file_name in file_names:
        file_path = os.path.join(base_path, file_name)
        ref_doc = webanno_tsv_read_file(file_path)
        predicted_doc = predictor(ref_doc)

        # Verify
        assert ref_doc.text == predicted_doc.text, 'content changed'
        assert len(ref_doc.sentences) == len(predicted_doc.sentences), 'sentences changed'
        assert len(ref_doc.tokens) == len(predicted_doc.tokens), 'tokens changed'
        for s1, s2 in zip(ref_doc.sentences, predicted_doc.sentences):
            assert s1 == s2, f'sentence changed, \n{s1}\n{s2}'

        for t1, t2 in zip(ref_doc.tokens, predicted_doc.tokens):
            assert t1 == t2, f'token changed: \n{t1}\n{t2}'

        print(f"Predicted {len(predicted_doc.annotations)} annotations")
        prediction_path = os.path.join(output_folder, file_name)
        with open(prediction_path, 'w') as fd:
            fd.write(predicted_doc.tsv())
