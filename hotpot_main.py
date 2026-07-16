import os
from prepro import prepro
from run import train, test
import argparse
import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self._chunks = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style') and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._chunks.append(text)

    def get_text(self):
        return ' '.join(self._chunks)


def _clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _safe_urlopen(url, timeout=10, headers=None):
    request = urllib.request.Request(url, headers=headers or {
        'User-Agent': 'hotpotqa-internet-retriever/1.0'
    })
    return urllib.request.urlopen(request, timeout=timeout)


def fetch_url_text(url, timeout=10, max_chars=5000):
    try:
        response = _safe_urlopen(url, timeout=timeout)
        content_type = response.headers.get('Content-Type', '')
        raw = response.read()
        charset = response.headers.get_content_charset() or 'utf-8'
        text = raw.decode(charset, errors='replace')

        if 'html' in content_type:
            parser = _HTMLTextExtractor()
            parser.feed(text)
            text = parser.get_text()

        text = _clean_text(text)
        if max_chars is not None:
            text = text[:max_chars]
        return text
    except Exception:
        return ''


def wikipedia_search(query, top_k=5):
    encoded = urllib.parse.quote(query)
    url = 'https://en.wikipedia.org/w/api.php?action=opensearch&search={}&limit={}&namespace=0&format=json'.format(encoded, top_k)
    try:
        response = _safe_urlopen(url)
        data = json.loads(response.read().decode('utf-8'))
        titles = data[1]
        descriptions = data[2]
        links = data[3]
        results = []
        for idx, title in enumerate(titles):
            results.append({
                'title': title,
                'snippet': descriptions[idx] if idx < len(descriptions) else '',
                'url': links[idx] if idx < len(links) else ''
            })
        return results
    except Exception:
        return []


def wikipedia_page_extract(title, max_chars=3000):
    encoded = urllib.parse.quote(title)
    url = 'https://en.wikipedia.org/api/rest_v1/page/summary/{}'.format(encoded)
    try:
        response = _safe_urlopen(url)
        data = json.loads(response.read().decode('utf-8'))
        extract = data.get('extract', '')
        return _clean_text(extract)[:max_chars]
    except Exception:
        return ''


def duckduckgo_search(query, top_k=5):
    encoded = urllib.parse.quote(query)
    url = 'https://duckduckgo.com/html/?q={}'.format(encoded)
    try:
        html = fetch_url_text(url, timeout=10, max_chars=20000)
        if not html:
            return []

        pattern = re.compile(r'(https?://[^\s]+)')
        urls = []
        for match in pattern.findall(html):
            if match not in urls:
                urls.append(match)
            if len(urls) >= top_k:
                break

        results = []
        for item in urls:
            results.append({
                'title': item,
                'snippet': '',
                'url': item
            })
        return results
    except Exception:
        return []


class InternetRetriever(object):
    def __init__(self, use_wikipedia=True, use_duckduckgo=False, fetch_pages=True,
                 top_k=5, snippet_chars=1200, page_chars=3000):
        self.use_wikipedia = use_wikipedia
        self.use_duckduckgo = use_duckduckgo
        self.fetch_pages = fetch_pages
        self.top_k = top_k
        self.snippet_chars = snippet_chars
        self.page_chars = page_chars

    def search(self, query):
        results = []
        seen = set()

        if self.use_wikipedia:
            wiki_results = wikipedia_search(query, top_k=self.top_k)
            for item in wiki_results:
                url = item.get('url', '')
                if url and url not in seen:
                    item['source'] = 'wikipedia'
                    results.append(item)
                    seen.add(url)

        if self.use_duckduckgo:
            web_results = duckduckgo_search(query, top_k=self.top_k)
            for item in web_results:
                url = item.get('url', '')
                if url and url not in seen:
                    item['source'] = 'duckduckgo'
                    results.append(item)
                    seen.add(url)

        return results[:self.top_k]

    def retrieve(self, query):
        results = self.search(query)
        enriched = []
        for item in results:
            text = item.get('snippet', '') or ''
            if self.fetch_pages:
                if item.get('source') == 'wikipedia' and item.get('title'):
                    page_text = wikipedia_page_extract(item['title'], max_chars=self.page_chars)
                else:
                    page_text = fetch_url_text(item.get('url', ''), max_chars=self.page_chars)
                if page_text:
                    text = page_text
            text = _clean_text(text)[:self.snippet_chars]
            enriched.append({
                'title': item.get('title', ''),
                'url': item.get('url', ''),
                'source': item.get('source', ''),
                'text': text
            })
        return enriched

    def format_context(self, query):
        docs = self.retrieve(query)
        lines = []
        for idx, doc in enumerate(docs):
            lines.append('[{}] {} ({})'.format(idx + 1, doc['title'], doc['source']))
            if doc['url']:
                lines.append('URL: {}'.format(doc['url']))
            lines.append(doc['text'])
            lines.append('')
        return '\n'.join(lines).strip()


def build_internet_retriever_from_config(config):
    return InternetRetriever(
        use_wikipedia=not config.disable_wikipedia_retriever,
        use_duckduckgo=config.enable_duckduckgo_retriever,
        fetch_pages=not config.disable_page_fetch,
        top_k=config.internet_top_k,
        snippet_chars=config.internet_snippet_chars,
        page_chars=config.internet_page_chars,
    )


def maybe_preview_internet_context(config):
    if not config.internet_retriever:
        return
    if not config.internet_query:
        print('Internet retriever enabled, but no --internet_query was provided.')
        return

    retriever = build_internet_retriever_from_config(config)
    context = retriever.format_context(config.internet_query)

    if config.internet_output_file:
        with open(config.internet_output_file, 'w') as fh:
            fh.write(context)
        print('Saved internet retrieval context to {}'.format(config.internet_output_file))
    else:
        print('Internet retrieval context:')
        print(context)


parser = argparse.ArgumentParser()

glove_word_file = 'glove.840B.300d.txt'

word_emb_file = 'word_emb.json'
char_emb_file = 'char_emb.json'
train_eval = 'train_eval.json'
dev_eval = 'dev_eval.json'
test_eval = 'test_eval.json'
word2idx_file = 'word2idx.json'
char2idx_file = 'char2idx.json'
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

parser.add_argument('--internet_retriever', action='store_true')
parser.add_argument('--internet_query', type=str, default=None)
parser.add_argument('--internet_output_file', type=str, default=None)
parser.add_argument('--internet_top_k', type=int, default=5)
parser.add_argument('--internet_snippet_chars', type=int, default=1200)
parser.add_argument('--internet_page_chars', type=int, default=3000)
parser.add_argument('--enable_duckduckgo_retriever', action='store_true')
parser.add_argument('--disable_wikipedia_retriever', action='store_true')
parser.add_argument('--disable_page_fetch', action='store_true')

config = parser.parse_args()


def _concat(filename):
    if config.fullwiki:
        return 'fullwiki.{}'.format(filename)
    return filename


config.dev_record_file = _concat(config.dev_record_file)
config.test_record_file = _concat(config.test_record_file)
config.dev_eval_file = _concat(config.dev_eval_file)
config.test_eval_file = _concat(config.test_eval_file)

if config.internet_retriever and config.mode == 'retrieve':
    maybe_preview_internet_context(config)
elif config.mode == 'train':
    train(config)
elif config.mode == 'prepro':
    prepro(config)
elif config.mode == 'test':
    if config.internet_retriever:
        maybe_preview_internet_context(config)
    test(config)
