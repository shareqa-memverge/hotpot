import os
import json
import re
import urllib.parse
import urllib.request
from prepro import prepro
from run import train, test
import argparse

parser = argparse.ArgumentParser()

glove_word_file = "glove.840B.300d.txt"
word_emb_file = "word_emb.json"
char_emb_file = "char_emb.json"
train_eval = "train_eval.json"
dev_eval = "dev_eval.json"
test_eval = "test_eval.json"
word2idx_file = "word2idx.json"
char2idx_file = "char2idx.json"
idx2word_file = 'idx2word.json'
idx2char_file = 'idx2char.json'
train_record_file = 'train_record.pkl'
dev_record_file = 'dev_record.pkl'
test_record_file = 'test_record.pkl'

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--data_file', type=str)
parser.add_argument('--glove_word_file', type=str, default=glove_word_file)
parser.add_argument('--save', type=str, default='HOTPOT')
parser.add_argument('--word_emb_file', type=str, default=word_emb_file)
parser.add_argument('--char_emb_file', type=str, default=char_emb_file)
parser.add_argument('--train_eval_file', type=str, default=train_eval)
parser.add_argument('--dev_eval_file', type=str, default=dev_eval)
parser.add_argument('--test_eval_file', type=str, default=test_eval)
parser.add_argument('--word2idx_file', type=str, default=word2idx_file)
parser.add_argument('--char2idx_file', type=str, default=char2idx_file)
parser.add_argument('--idx2word_file', type=str, default=idx2word_file)
parser.add_argument('--idx2char_file', type=str, default=idx2char_file)
parser.add_argument('--train_record_file', type=str, default=train_record_file)
parser.add_argument('--dev_record_file', type=str, default=dev_record_file)
parser.add_argument('--test_record_file', type=str, default=test_record_file)
parser.add_argument('--glove_char_size', type=int, default=94)
parser.add_argument('--glove_word_size', type=int, default=int(2.2e6))
parser.add_argument('--glove_dim', type=int, default=300)
parser.add_argument('--char_dim', type=int, default=8)
parser.add_argument('--para_limit', type=int, default=1000)
parser.add_argument('--ques_limit', type=int, default=80)
parser.add_argument('--sent_limit', type=int, default=100)
parser.add_argument('--char_limit', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--checkpoint', type=int, default=1000)
parser.add_argument('--period', type=int, default=100)
parser.add_argument('--init_lr', type=float, default=0.5)
parser.add_argument('--keep_prob', type=float, default=0.8)
parser.add_argument('--hidden', type=int, default=80)
parser.add_argument('--char_hidden', type=int, default=100)
parser.add_argument('--patience', type=int, default=1)
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--sp_lambda', type=float, default=0.0)
parser.add_argument('--data_split', type=str, default='train')
parser.add_argument('--fullwiki', action='store_true')
parser.add_argument('--prediction_file', type=str)
parser.add_argument('--sp_threshold', type=float, default=0.3)
parser.add_argument('--question', type=str)
parser.add_argument('--retriever_output', type=str, default='retrieved_context.json')
parser.add_argument('--retriever_topk', type=int, default=10)
parser.add_argument('--retriever_timeout', type=float, default=10.0)


def _http_get_json(url, timeout=10.0):
    request = urllib.request.Request(url, headers={'User-Agent': 'HotpotQA-internet-retriever/1.0'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def wikipedia_search(query, top_k=10, timeout=10.0):
    params = urllib.parse.urlencode({'action': 'query', 'list': 'search', 'srsearch': query, 'srlimit': max(1, int(top_k)), 'format': 'json', 'utf8': 1})
    data = _http_get_json('https://en.wikipedia.org/w/api.php?' + params, timeout)
    return [item['title'] for item in data.get('query', {}).get('search', [])]


def wikipedia_page(title, timeout=10.0):
    params = urllib.parse.urlencode({'action': 'query', 'prop': 'extracts', 'explaintext': 1, 'exsectionformat': 'plain', 'titles': title, 'format': 'json', 'redirects': 1})
    data = _http_get_json('https://en.wikipedia.org/w/api.php?' + params, timeout)
    page = next(iter(data.get('query', {}).get('pages', {}).values()), {})
    extract = page.get('extract', '')
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', extract) if s.strip()]
    return [page.get('title', title), sentences] if sentences else None


def internet_retrieve(question, top_k=10, timeout=10.0):
    context = []
    for title in wikipedia_search(question, top_k, timeout):
        try:
            page = wikipedia_page(title, timeout)
            if page:
                context.append(page)
        except (IOError, ValueError, KeyError):
            continue
    return context


def retrieve_questions(input_file, output_file, top_k=10, timeout=10.0):
    with open(input_file, 'r') as handle:
        records = json.load(handle)
    for record in records:
        record['context'] = internet_retrieve(record['question'], top_k, timeout)
    with open(output_file, 'w') as handle:
        json.dump(records, handle, indent=2)
    return records


config = parser.parse_args()

def _concat(filename):
    if config.fullwiki:
        return 'fullwiki.{}'.format(filename)
    return filename
config.dev_record_file = _concat(config.dev_record_file)
config.test_record_file = _concat(config.test_record_file)
config.dev_eval_file = _concat(config.dev_eval_file)
config.test_eval_file = _concat(config.test_eval_file)

if config.mode == 'train':
    train(config)
elif config.mode == 'prepro':
    prepro(config)
elif config.mode == 'test':
    test(config)
elif config.mode == 'count':
    cnt_len(config)
elif config.mode == 'retrieve':
    if not config.data_file:
        parser.error('--data_file is required when --mode retrieve')
    retrieve_questions(config.data_file, config.retriever_output, config.retriever_topk, config.retriever_timeout)
elif config.mode == 'retrieve_question':
    if not config.question:
        parser.error('--question is required when --mode retrieve_question')
    print(json.dumps(internet_retrieve(config.question, config.retriever_topk, config.retriever_timeout), indent=2))
