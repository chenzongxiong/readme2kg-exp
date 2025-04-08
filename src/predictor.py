import random
from collections import defaultdict
from webanno_tsv import webanno_tsv_read_file, Document, Annotation, Token
import utils

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


class BasePredictor:
    def __call__(self, doc: Document):
        annotations = []
        for sent in doc.sentences:
            tokens = doc.sentence_tokens(sent)
            # NOTE: PUT YOUR PREDICTION LOGIC HERE

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


class DummpyPredictor(BasePredictor):
    def predict(self, sentence, tokens):
        '''
        Only predict one label for each sentence
        '''
        start_char = random.randint(tokens[0].start, tokens[-1].end - 1)
        end_char = random.randint(start_char + 1, tokens[-1].end)
        # NOTE: Since our NER task is character-level, we need handle the tokens at the edge of span carefully.
        span_tokens = utils.make_span_tokens(tokens, start_char, end_char)
        if span_tokens is None:
            return []
        # Random select a label from LABELS
        label = random.choice(LABELS)
        return [{'span_tokens': span_tokens, 'label': label}]


if __name__ == "__main__":
    import os
    base_path = './data/train'
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
