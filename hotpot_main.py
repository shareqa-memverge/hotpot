"""
hotpot_main.py

This module extends the original HotpotQA `main.py` entry point with an
internet retriever that can fetch supporting documents/paragraphs for a
question from the web, in the same `context` format (`[title, [sentences]]`)
that `prepro.py` expects (see `_process_article` in prepro.py).

It supports two retrieval backends:
  1. Wikipedia (via the public Wikipedia REST/MediaWiki API, no key needed).
  2. Generic web search + page scraping (via a pluggable search function,
     e.g. Bing/Google/SerpAPI, if an API key is configured through env vars).

Usage (standalone):
    python hotpot_main.py --mode retrieve --question "Who is the mayor of X?" \
        --save_file retrieved.json

Usage (as a drop-in replacement for main.py's train/prepro/test modes):
    python hotpot_main.py --mode train ...
    python hotpot_main.py --mode prepro ...
    python hotpot_main.py --mode test ...

The retriever additions do not change any existing behavior of main.py;
they only add new functions and a new `--mode retrieve` / `--mode
fullwiki_retrieve` pathway.
"""

import os
import json as _json
import time
import argparse
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from prepro import prepro
from run import train, test

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

# ---------------------------------------------------------------------------
# New arguments for internet retriever functionality.
# ---------------------------------------------------------------------------
parser.add_argument('--question', type=str, default=None,
                     help='Question to retrieve supporting documents for '
                          '(used with --mode retrieve/fullwiki_retrieve).')
parser.add_argument('--retriever', type=str, default='wikipedia',
                     choices=['wikipedia', 'web'],
                     help='Which internet retriever backend to use.')
parser.add_argument('--num_docs', type=int, default=5,
                     help='Number of documents/paragraphs to retrieve.')
parser.add_argument('--save_file', type=str, default='retrieved.json',
                     help='Where to write retrieved context / full examples.')
parser.add_argument('--retrieve_timeout', type=float, default=10.0,
                     help='HTTP timeout (seconds) for retriever requests.')


# ===========================================================================
# Internet retriever functionality
# ===========================================================================
#
# These functions fetch content from the internet and reshape it into the
# `context` format used throughout this codebase:
#
#     context = [[title_1, [sent_1, sent_2, ...]], [title_2, [...]], ...]
#
# which is exactly what `_process_article` in prepro.py consumes for the
# `article['context']` field, so retrieved results can be fed straight into
# the existing preprocessing / training / testing pipeline.

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
_WHITESPACE_RE = re.compile(r'\s+')
_USER_AGENT = 'hotpotqa-hotpot-main/1.0 (internet retriever)'


def _http_get_json(url, timeout=10.0):
    """GET a URL and parse the response body as JSON."""
    req = urllib.request.Request(url, headers={'User-Agent': _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return _json.loads(body.decode('utf-8', errors='replace'))


def _http_get_text(url, timeout=10.0):
    """GET a URL and return the response body decoded as text."""
    req = urllib.request.Request(url, headers={'User-Agent': _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or 'utf-8'
        body = resp.read()
    return body.decode(charset, errors='replace')


def split_into_sentences(text):
    """Very lightweight sentence splitter used to shape retrieved text into
    the per-sentence list format expected by the HotpotQA context schema."""
    text = _WHITESPACE_RE.sub(' ', text).strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


class _TextExtractingHTMLParser(HTMLParser):
    """Minimal HTML-to-text extractor with no third-party dependencies, used
    as a fallback for the generic web retriever when scraping raw pages."""

    _SKIP_TAGS = {'script', 'style', 'noscript', 'head', 'nav', 'footer'}

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self.chunks = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            data = data.strip()
            if data:
                self.chunks.append(data)

    def get_text(self):
        return ' '.join(self.chunks)


def html_to_text(html):
    parser_ = _TextExtractingHTMLParser()
    try:
        parser_.feed(html)
    except Exception:
        pass
    return parser_.get_text()


def wikipedia_search_titles(query, limit=5, timeout=10.0):
    """Use the Wikipedia search API to find candidate page titles for a
    natural-language query (e.g. a HotpotQA question)."""
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': query,
        'srlimit': limit,
        'format': 'json',
    }
    url = 'https://en.wikipedia.org/w/api.php?' + urllib.parse.urlencode(params)
    data = _http_get_json(url, timeout=timeout)
    results = data.get('query', {}).get('search', [])
    return [r['title'] for r in results]


def wikipedia_fetch_extract(title, timeout=10.0):
    """Fetch the plain-text extract (intro + body, HTML stripped) of a
    Wikipedia page by title, using the MediaWiki `extracts` API."""
    params = {
        'action': 'query',
        'prop': 'extracts',
        'explaintext': 1,
        'titles': title,
        'format': 'json',
    }
    url = 'https://en.wikipedia.org/w/api.php?' + urllib.parse.urlencode(params)
    data = _http_get_json(url, timeout=timeout)
    pages = data.get('query', {}).get('pages', {})
    for _, page in pages.items():
        if 'extract' in page:
            return page.get('title', title), page['extract']
    return title, ''


def retrieve_wikipedia_context(question, num_docs=5, timeout=10.0):
    """Internet retriever: given a question, search Wikipedia and return a
    `context` list in the HotpotQA format:

        [[title_1, [sent_1, sent_2, ...]], [title_2, [...]], ...]

    This can be plugged directly into an article dict's `context` field
    before running it through `prepro.process_file` / `_process_article`.
    """
    titles = wikipedia_search_titles(question, limit=num_docs, timeout=timeout)
    context = []
    for title in titles:
        try:
            fetched_title, extract = wikipedia_fetch_extract(title, timeout=timeout)
        except Exception as e:
            print('Warning: failed to fetch Wikipedia page "{}": {}'.format(title, e))
            continue
        sentences = split_into_sentences(extract)
        if sentences:
            context.append([fetched_title, sentences])
    return context


def web_search(query, num_results=5, timeout=10.0):
    """Generic web search used by the `web` retriever backend.

    Looks for a configured search API key in the environment so that users
    can plug in their preferred provider without modifying this file:
      - BING_SEARCH_API_KEY -> Bing Web Search API
      - SERPAPI_API_KEY     -> SerpAPI (Google results)

    Returns a list of result URLs. If no API key is configured, falls back
    to DuckDuckGo's HTML endpoint (no key required, best-effort only).
    """
    bing_key = os.environ.get('BING_SEARCH_API_KEY')
    serpapi_key = os.environ.get('SERPAPI_API_KEY')

    if bing_key:
        url = 'https://api.bing.microsoft.com/v7.0/search?' + urllib.parse.urlencode({
            'q': query, 'count': num_results,
        })
        req = urllib.request.Request(url, headers={
            'User-Agent': _USER_AGENT,
            'Ocp-Apim-Subscription-Key': bing_key,
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = _json.loads(resp.read().decode('utf-8', errors='replace'))
        return [r['url'] for r in data.get('webPages', {}).get('value', [])]

    if serpapi_key:
        url = 'https://serpapi.com/search.json?' + urllib.parse.urlencode({
            'q': query, 'num': num_results, 'api_key': serpapi_key,
        })
        data = _http_get_json(url, timeout=timeout)
        return [r['link'] for r in data.get('organic_results', []) if 'link' in r]

    # Fallback: DuckDuckGo HTML search (no API key required).
    url = 'https://html.duckduckgo.com/html/?' + urllib.parse.urlencode({'q': query})
    try:
        html = _http_get_text(url, timeout=timeout)
    except Exception as e:
        print('Warning: web_search fallback failed: {}'.format(e))
        return []
    urls = re.findall(r'href="(https?://[^"]+)"', html)
    seen, deduped = set(), []
    for u in urls:
        if 'duckduckgo.com' in u:
            continue
        if u not in seen:
            seen.add(u)
            deduped.append(u)
        if len(deduped) >= num_results:
            break
    return deduped


def fetch_page_text(url, timeout=10.0):
    """Download a URL and return extracted plain text."""
    html = _http_get_text(url, timeout=timeout)
    return html_to_text(html)


def page_title_from_url(url):
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.strip('/').split('/')[-1] or parsed.netloc
    return urllib.parse.unquote(path.replace('-', ' ').replace('_', ' ')) or url


def retrieve_web_context(question, num_docs=5, timeout=10.0):
    """Internet retriever: run a generic web search for `question`, fetch
    each result page, and shape the extracted text into the HotpotQA
    `context` format: [[title_1, [sent_1, ...]], ...].
    """
    urls = web_search(question, num_results=num_docs, timeout=timeout)
    context = []
    for url in urls:
        try:
            text = fetch_page_text(url, timeout=timeout)
        except Exception as e:
            print('Warning: failed to fetch "{}": {}'.format(url, e))
            continue
        sentences = split_into_sentences(text)
        if sentences:
            context.append([page_title_from_url(url), sentences])
    return context


def retrieve_context(question, retriever='wikipedia', num_docs=5, timeout=10.0):
    """Dispatch to the configured internet retriever backend and return a
    HotpotQA-style `context` list for `question`."""
    if retriever == 'wikipedia':
        return retrieve_wikipedia_context(question, num_docs=num_docs, timeout=timeout)
    elif retriever == 'web':
        return retrieve_web_context(question, num_docs=num_docs, timeout=timeout)
    else:
        raise ValueError('Unknown retriever backend: {}'.format(retriever))


def build_retrieved_article(question, qid='retrieved_0', retriever='wikipedia',
                             num_docs=5, timeout=10.0):
    """Build a single HotpotQA-style article dict (without gold answer or
    supporting_facts, since those are unknown for live, retrieved data) that
    can be fed straight into `prepro._process_article` / `process_file` for
    full-wiki style open-domain inference.
    """
    context = retrieve_context(question, retriever=retriever, num_docs=num_docs,
                                timeout=timeout)
    return {
        '_id': qid,
        'question': question,
        'context': context,
        'supporting_facts': [],
    }


def retrieve_and_save(question, save_file, retriever='wikipedia', num_docs=5,
                       timeout=10.0):
    """Retrieve context for `question` from the internet and save a single-
    article JSON file in the HotpotQA fullwiki data format, ready to be
    consumed by `--mode prepro --fullwiki --data_file save_file`.
    """
    article = build_retrieved_article(question, retriever=retriever,
                                       num_docs=num_docs, timeout=timeout)
    with open(save_file, 'w', encoding='utf-8') as f:
        _json.dump([article], f, ensure_ascii=False, indent=2)
    print('Retrieved {} paragraph(s) for question "{}" using "{}" retriever.'.format(
        len(article['context']), question, retriever))
    print('Saved to {}'.format(save_file))
    return article


def retrieve_batch_and_save(data_file, save_file, retriever='wikipedia', num_docs=5,
                             timeout=10.0, sleep_between=0.0):
    """Batch version of `retrieve_and_save`: reads a JSON file containing a
    list of HotpotQA-style articles (only the `question`/`_id` fields are
    required), retrieves fresh internet context for each question, and
    writes out a new file with the `context` field replaced/filled in.

    This is useful for building a fullwiki-style dataset where paragraphs
    come from a live internet retriever rather than a pre-built index.
    """
    with open(data_file, 'r', encoding='utf-8') as f:
        articles = _json.load(f)

    retrieved_articles = []
    for i, article in enumerate(articles):
        question = article['question']
        qid = article.get('_id', 'retrieved_{}'.format(i))
        try:
            context = retrieve_context(question, retriever=retriever,
                                        num_docs=num_docs, timeout=timeout)
        except Exception as e:
            print('Warning: retrieval failed for "{}" ({}): {}'.format(qid, question, e))
            context = []
        new_article = dict(article)
        new_article['context'] = context
        new_article.setdefault('supporting_facts', [])
        retrieved_articles.append(new_article)
        print('[{}/{}] retrieved {} paragraph(s) for "{}"'.format(
            i + 1, len(articles), len(context), question))
        if sleep_between > 0:
            time.sleep(sleep_between)

    with open(save_file, 'w', encoding='utf-8') as f:
        _json.dump(retrieved_articles, f, ensure_ascii=False, indent=2)
    print('Saved {} retrieved article(s) to {}'.format(len(retrieved_articles), save_file))
    return retrieved_articles


# ===========================================================================
# Entry point
# ===========================================================================

config = parser.parse_args()


def _concat(filename):
    if config.fullwiki:
        return 'fullwiki.{}'.format(filename)
    return filename
# config.train_record_file = _concat(config.train_record_file)
config.dev_record_file = _concat(config.dev_record_file)
config.test_record_file = _concat(config.test_record_file)
# config.train_eval_file = _concat(config.train_eval_file)
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
    # Single-question internet retrieval, e.g.:
    #   python hotpot_main.py --mode retrieve --question "..." \
    #       --retriever wikipedia --num_docs 5 --save_file retrieved.json
    if not config.question:
        raise ValueError('--question is required for --mode retrieve')
    retrieve_and_save(config.question, config.save_file,
                       retriever=config.retriever, num_docs=config.num_docs,
                       timeout=config.retrieve_timeout)
elif config.mode == 'fullwiki_retrieve':
    # Batch internet retrieval over an existing HotpotQA-format file
    # (only question/_id fields are needed), e.g.:
    #   python hotpot_main.py --mode fullwiki_retrieve \
    #       --data_file questions.json --retriever web --num_docs 5 \
    #       --save_file fullwiki.retrieved.json
    if not config.data_file:
        raise ValueError('--data_file is required for --mode fullwiki_retrieve')
    retrieve_batch_and_save(config.data_file, config.save_file,
                             retriever=config.retriever, num_docs=config.num_docs,
                             timeout=config.retrieve_timeout)
