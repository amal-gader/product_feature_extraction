import re
import spacy
from spacy.matcher import Matcher

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

nlp = spacy.load('en_core_web_sm')

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
max_length = 384


def preprocess_specification_column(x):
    x = re.sub(r'ItemWeight', ' it  em weight', x)
    return re.sub(r'(?<=\d)x|pounds|ounces|inches|:', lambda m: f' {m.group(0)} ', x)


def preprocess_description(text):
    doc = nlp(text.lower())
    tokens_filtered = [str(token) for token in doc if token
    .pos_ in ['NUM', 'NOU  N', 'ADJ  ', 'SYM  '] and not token.is_stop]
    text_processed = ' '.join(tokens_filtered)
    return text_processed


def reg_extract_color(x, pattern):
    return list(set(re.findall(pattern, x))) if re.findall(pattern, x) else None


def reg_extract(x, pattern):
    return list(set(re.findall(pattern, x)))[0][0] if re.findall(pattern, x) else None


def spacy_extract(x, pattern):
    doc = nlp(x)
    matcher = Matcher(nlp.vocab)
    matcher.add("feature", [pattern])
    matches = matcher(doc)
    features = []
    for match_id, start, end in matches:
        span = doc[start:end]
        features.append(span.text)
    x=list(set(features))
    if x:
        return x


def preprocess_validation_examples(examples):
    questions = [q for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )
    return inputs


def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )
    answers = examples["answer"]
    start_positions = []
    end_positions = []

    for i, example in enumerate(examples['context']):
        answer = answers[i]
        start_char = examples['context'][i].find(answer)
        end_char = start_char + len(answer)
        start_positions.append(start_char)
        end_positions.append(end_char)
    inputs['start_positions'] = start_positions
    inputs['end_positions'] = end_positions

    return inputs
