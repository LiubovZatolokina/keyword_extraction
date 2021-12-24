import numpy as np
from transformers import BertConfig, BertTokenizer, BertForTokenClassification
import torch

import re

config = BertConfig.from_json_file('bert_keywords/config.json')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForTokenClassification.from_pretrained("bert_keywords/pytorch_model.bin", config=config)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


def extract_keywords(sentence):
    text = re.sub('[^A-Za-z0-9]+', ' ', sentence).lower()
    tokens = tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=False)
    bert_model.eval()
    prediction = []
    logit = bert_model(tokens['input_ids'])
    logit = logit.logits.detach().cpu().numpy()
    prediction.extend([list(p) for p in np.argmax(logit, 2)])
    print(prediction)
    source = [tokenizer.decode(i, clean_up_tokenization_spaces=True, skip_special_tokens=True)
              for i in tokens['input_ids']]
    for k, j in enumerate(prediction[0][1:-1]):
        if j == 1 or j == 0:
            print(source[0].split()[k], j)


if __name__=='__main__':
    text = 'Keyword extraction (also known as keyword detection or keyword analysis) is a text analysis technique ' \
           'that automatically extracts the most used and most important words and expressions from a text. It helps ' \
           'summarize the content of texts and recognize the main topics discussed. '
    extract_keywords(text)