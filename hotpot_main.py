"""
hotpot_main.py
==============
Drop-in replacement / extension of main.py that adds a full Internet Retriever
module to the HotpotQA pipeline.

New capabilities
----------------
* WikipediaRetriever  – fetches intro paragraphs via the Wikipedia REST API
* WebSearchRetriever  – returns top-N snippets via DuckDuckGo Instant Answer API
* PageFetcher        – downloads and cleans arbitrary web pages
* InternetRetriever  – orchestrator: queries all back-ends, deduplicates results,
                       scores them by keyword overlap, and formats a single context
                       string ready for the HotpotQA model.

CLI flags added
---------------
--internet_retrieve          enable the internet retriever at test/prepro time
--ir_top_k         int=5     number of passages to keep after scoring
--ir_min_len       int=30    minimum character length for a snippet to be kept
--ir_timeout       float=5   HTTP request timeout in seconds
--ir_cache_dir     str=''    if set, cache raw retrieval results to this folder

Usage
-----
# train (unchanged)
python hotpot_main.py --mode train ...

# test with internet retrieval
python hotpot_main.py --mode test --internet_retrieve --ir_top_k 8 ...
"""

import os
import re
import json
import time
import hashlib
import logging
import argparse
import unicodedata
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup

from prepro import prepro
from run import train, test

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("hotpot_internet_retriever")


# ===========================================================================
# Internet Retriever Module
# ===========================================================================

class WikipediaRetriever:
    """
    Retrieves the introductory paragraphs of a Wikipedia article using the
    Wikipedia REST summary API (no API key required).

    API reference: https://en.wikipedia.org/api/rest_v1/#/Page%20content/get_page_summary__title_
    """

    BASE_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
    SEARCH_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "HotpotQA-InternetRetriever/1.0 (research project)"
        })

    def _search_titles(self, query: str, top_k: int = 3) -> List[str]:
        """Return up to *top_k* Wikipedia page titles matching *query*."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": top_k,
            "format": "json",
            "utf8": 1,
        }
        try:
            r = self.session.get(self.SEARCH_URL, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return [item["title"] for item in data.get("query", {}).get("search", [])]
        except Exception as exc:
            log.warning("Wikipedia title search failed: %s", exc)
            return []

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search Wikipedia for *query* and return a list of passage dicts::

            [
              {"title": str, "text": str, "url": str, "source": "wikipedia"},
              ...
            ]
        """
        titles = self._search_titles(query, top_k=top_k)
        passages = []
        for title in titles:
            try:
                url = self.BASE_URL.format(requests.utils.quote(title))
                r = self.session.get(url, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                text = data.get("extract", "").strip()
                if text:
                    passages.append({
                        "title": data.get("title", title),
                        "text": text,
                        "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                        "source": "wikipedia",
                    })
            except Exception as exc:
                log.warning("Wikipedia summary fetch failed for '%s': %s", title, exc)
        return passages


class WebSearchRetriever:
    """
    Returns top-N search result snippets using the DuckDuckGo Instant Answer
    API (no key required) and optionally fetches the first result page.

    Note: DuckDuckGo's instant-answer API is best-effort and not guaranteed for
    production use.  Replace the URL / parsing logic if you have access to a
    paid API (Google, Bing, Tavily, Serper, etc.).
    """

    DDG_URL = "https://api.duckduckgo.com/"

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "HotpotQA-InternetRetriever/1.0 (research project)"
        })

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Query DuckDuckGo and return passage dicts::

            [{"title": str, "text": str, "url": str, "source": "duckduckgo"}, ...]
        """
        params = {
            "q": query,
            "format": "json",
            "no_redirect": 1,
            "no_html": 1,
            "skip_disambig": 1,
        }
        passages = []
        try:
            r = self.session.get(self.DDG_URL, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()

            # Abstract (single definitive answer)
            abstract = data.get("AbstractText", "").strip()
            if abstract:
                passages.append({
                    "title": data.get("Heading", query),
                    "text": abstract,
                    "url": data.get("AbstractURL", ""),
                    "source": "duckduckgo",
                })

            # Related topics
            for topic in data.get("RelatedTopics", [])[:top_k]:
                # Topics can be nested groups
                if "Topics" in topic:
                    for sub in topic["Topics"][:2]:
                        text = sub.get("Text", "").strip()
                        if text:
                            passages.append({
                                "title": sub.get("FirstURL", "").split("/")[-1].replace("_", " "),
                                "text": text,
                                "url": sub.get("FirstURL", ""),
                                "source": "duckduckgo",
                            })
                else:
                    text = topic.get("Text", "").strip()
                    if text:
                        passages.append({
                            "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                            "text": text,
                            "url": topic.get("FirstURL", ""),
                            "source": "duckduckgo",
                        })
                if len(passages) >= top_k:
                    break
        except Exception as exc:
            log.warning("DuckDuckGo retrieval failed: %s", exc)

        return passages[:top_k]


class PageFetcher:
    """
    Fetches a single web page and extracts clean plain text from it.
    Useful when you already have a URL (e.g. from a search result) and want
    the full paragraph text rather than just the snippet.
    """

    def __init__(self, timeout: float = 8.0, max_chars: int = 4000):
        self.timeout = timeout
        self.max_chars = max_chars
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; HotpotQA-Retriever/1.0)"
        })

    @staticmethod
    def _clean(text: str) -> str:
        """Normalise whitespace and strip control characters."""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def fetch(self, url: str) -> Optional[str]:
        """
        Download *url* and return the first *max_chars* characters of the
        main body text, or None on failure.
        """
        if not url or not url.startswith("http"):
            return None
        try:
            r = self.session.get(url, timeout=self.timeout)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            # Remove boilerplate tags
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            text = " ".join(paragraphs)
            text = self._clean(text)
            return text[: self.max_chars] if text else None
        except Exception as exc:
            log.warning("PageFetcher failed for '%s': %s", url, exc)
            return None


class InternetRetriever:
    """
    Orchestrator that queries Wikipedia and DuckDuckGo, deduplicates results,
    scores passages by keyword overlap with the original question, and returns
    the top-K passages formatted as a single context string.

    Parameters
    ----------
    top_k : int
        Maximum number of passages to return after scoring.
    min_len : int
        Passages shorter than this (characters) are discarded.
    timeout : float
        Shared HTTP timeout for all sub-retrievers.
    cache_dir : str or None
        If given, raw retrieval results are cached to JSON files in this
        directory so repeat queries skip the network.
    use_page_fetcher : bool
        If True, fetch the full page for any URL that scores in the top-K.
    """

    def __init__(
        self,
        top_k: int = 5,
        min_len: int = 30,
        timeout: float = 5.0,
        cache_dir: Optional[str] = None,
        use_page_fetcher: bool = False,
    ):
        self.top_k = top_k
        self.min_len = min_len
        self.cache_dir = cache_dir
        self.use_page_fetcher = use_page_fetcher

        self._wiki = WikipediaRetriever(timeout=timeout)
        self._web = WebSearchRetriever(timeout=timeout)
        self._fetcher = PageFetcher(timeout=timeout + 3) if use_page_fetcher else None

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------

    def _cache_key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def _load_cache(self, query: str) -> Optional[List[Dict]]:
        if not self.cache_dir:
            return None
        path = os.path.join(self.cache_dir, self._cache_key(query) + ".json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    log.debug("Cache hit for query: %s", query)
                    return json.load(f)
            except Exception:
                pass
        return None

    def _save_cache(self, query: str, passages: List[Dict]) -> None:
        if not self.cache_dir:
            return
        path = os.path.join(self.cache_dir, self._cache_key(query) + ".json")
        try:
            with open(path, "w") as f:
                json.dump(passages, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            log.warning("Cache write failed: %s", exc)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> set:
        """Lowercase word tokens, stripping punctuation."""
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def _score(self, passage: Dict, query_tokens: set) -> float:
        """
        Jaccard-like overlap between query tokens and passage tokens.
        Wikipedia passages get a small bonus because they tend to be
        more reliable.
        """
        p_tokens = self._tokenize(passage["text"])
        if not p_tokens:
            return 0.0
        overlap = len(query_tokens & p_tokens) / (len(query_tokens | p_tokens) + 1e-9)
        bonus = 0.05 if passage.get("source") == "wikipedia" else 0.0
        return overlap + bonus

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup(passages: List[Dict]) -> List[Dict]:
        """
        Remove passages whose first 80 characters are identical (case-insensitive).
        """
        seen = set()
        unique = []
        for p in passages:
            key = p["text"][:80].lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, question: str) -> List[Dict]:
        """
        Retrieve passages for *question*.

        Returns a list of passage dicts, each containing:
            title   : str  – document / page title
            text    : str  – passage text
            url     : str  – source URL (may be empty)
            source  : str  – "wikipedia" | "duckduckgo"
            score   : float – relevance score
        """
        # 1. Cache lookup
        cached = self._load_cache(question)
        if cached is not None:
            return cached

        # 2. Query both back-ends
        wiki_passages = self._wiki.retrieve(question, top_k=self.top_k)
        web_passages = self._web.retrieve(question, top_k=self.top_k)
        all_passages = wiki_passages + web_passages

        # 3. Filter short passages
        all_passages = [p for p in all_passages if len(p["text"]) >= self.min_len]

        # 4. Optionally enrich with full-page text
        if self._fetcher:
            for p in all_passages:
                if p.get("url"):
                    full_text = self._fetcher.fetch(p["url"])
                    if full_text and len(full_text) > len(p["text"]):
                        p["text"] = full_text

        # 5. Deduplicate
        all_passages = self._dedup(all_passages)

        # 6. Score and sort
        query_tokens = self._tokenize(question)
        for p in all_passages:
            p["score"] = self._score(p, query_tokens)
        all_passages.sort(key=lambda x: x["score"], reverse=True)

        # 7. Keep top-K
        top_passages = all_passages[: self.top_k]

        # 8. Cache result
        self._save_cache(question, top_passages)

        return top_passages

    def format_context(self, question: str) -> str:
        """
        High-level helper: retrieve passages and return a single context
        string formatted for injection into the HotpotQA input.

        Each passage is prefixed with its index and title::

            [1] (Wikipedia) Marie Curie\n<text>\n\n
            [2] (DuckDuckGo) Polonium\n<text>\n\n
            ...
        """
        passages = self.retrieve(question)
        if not passages:
            log.info("Internet retriever returned no results for: %s", question)
            return ""

        lines = []
        for i, p in enumerate(passages, start=1):
            source_label = p.get("source", "web").capitalize()
            title = p.get("title", "") or "Unknown"
            lines.append(f"[{i}] ({source_label}) {title}")
            lines.append(p["text"])
            lines.append("")
        return "\n".join(lines).strip()


# ===========================================================================
# CLI argument parsing  (all original args preserved + new internet-retriever
# flags)
# ===========================================================================

parser = argparse.ArgumentParser(
    description="HotpotQA main entry-point with Internet Retriever support."
)

# ---- original file paths --------------------------------------------------
glove_word_file    = "glove.840B.300d.txt"
word_emb_file      = "word_emb.json"
char_emb_file      = "char_emb.json"
train_eval         = "train_eval.json"
dev_eval           = "dev_eval.json"
test_eval          = "test_eval.json"
word2idx_file      = "word2idx.json"
char2idx_file      = "char2idx.json"
idx2word_file      = "idx2word.json"
idx2char_file      = "idx2char.json"
train_record_file  = "train_record.pkl"
dev_record_file    = "dev_record.pkl"
test_record_file   = "test_record.pkl"

# ---- original args (unchanged) --------------------------------------------
parser.add_argument("--mode",             type=str,   default="train")
parser.add_argument("--data_file",        type=str)
parser.add_argument("--glove_word_file",  type=str,   default=glove_word_file)
parser.add_argument("--save",             type=str,   default="HOTPOT")

parser.add_argument("--word_emb_file",    type=str,   default=word_emb_file)
parser.add_argument("--char_emb_file",    type=str,   default=char_emb_file)
parser.add_argument("--train_eval_file",  type=str,   default=train_eval)
parser.add_argument("--dev_eval_file",    type=str,   default=dev_eval)
parser.add_argument("--test_eval_file",   type=str,   default=test_eval)
parser.add_argument("--word2idx_file",    type=str,   default=word2idx_file)
parser.add_argument("--char2idx_file",    type=str,   default=char2idx_file)
parser.add_argument("--idx2word_file",    type=str,   default=idx2word_file)
parser.add_argument("--idx2char_file",    type=str,   default=idx2char_file)

parser.add_argument("--train_record_file", type=str,  default=train_record_file)
parser.add_argument("--dev_record_file",   type=str,  default=dev_record_file)
parser.add_argument("--test_record_file",  type=str,  default=test_record_file)

parser.add_argument("--glove_char_size",  type=int,   default=94)
parser.add_argument("--glove_word_size",  type=int,   default=int(2.2e6))
parser.add_argument("--glove_dim",        type=int,   default=300)
parser.add_argument("--char_dim",         type=int,   default=8)

parser.add_argument("--para_limit",       type=int,   default=1000)
parser.add_argument("--ques_limit",       type=int,   default=80)
parser.add_argument("--sent_limit",       type=int,   default=100)
parser.add_argument("--char_limit",       type=int,   default=16)

parser.add_argument("--batch_size",       type=int,   default=64)
parser.add_argument("--checkpoint",       type=int,   default=1000)
parser.add_argument("--period",           type=int,   default=100)
parser.add_argument("--init_lr",          type=float, default=0.5)
parser.add_argument("--keep_prob",        type=float, default=0.8)
parser.add_argument("--hidden",           type=int,   default=80)
parser.add_argument("--char_hidden",      type=int,   default=100)
parser.add_argument("--patience",         type=int,   default=1)
parser.add_argument("--seed",             type=int,   default=13)

parser.add_argument("--sp_lambda",        type=float, default=0.0)
parser.add_argument("--data_split",       type=str,   default="train")
parser.add_argument("--fullwiki",         action="store_true")
parser.add_argument("--prediction_file",  type=str)
parser.add_argument("--sp_threshold",     type=float, default=0.3)

# ---- NEW: internet retriever flags ----------------------------------------
parser.add_argument(
    "--internet_retrieve",
    action="store_true",
    help="Enable internet retrieval (Wikipedia + DuckDuckGo) at test/prepro time.",
)
parser.add_argument(
    "--ir_top_k",
    type=int,
    default=5,
    help="Number of passages to keep after scoring (default: 5).",
)
parser.add_argument(
    "--ir_min_len",
    type=int,
    default=30,
    help="Minimum passage character length (default: 30).",
)
parser.add_argument(
    "--ir_timeout",
    type=float,
    default=5.0,
    help="HTTP request timeout in seconds for all retrievers (default: 5.0).",
)
parser.add_argument(
    "--ir_cache_dir",
    type=str,
    default="",
    help="Directory to cache raw retrieval results. Empty string disables caching.",
)
parser.add_argument(
    "--ir_use_page_fetcher",
    action="store_true",
    help="Fetch full page text for each retrieved URL to enrich snippets.",
)

config = parser.parse_args()


# ===========================================================================
# Fullwiki filename helpers  (unchanged from original main.py)
# ===========================================================================

def _concat(filename: str) -> str:
    if config.fullwiki:
        return "fullwiki.{}".format(filename)
    return filename


config.dev_record_file  = _concat(config.dev_record_file)
config.test_record_file = _concat(config.test_record_file)
config.dev_eval_file    = _concat(config.dev_eval_file)
config.test_eval_file   = _concat(config.test_eval_file)


# ===========================================================================
# Build the InternetRetriever if requested
# ===========================================================================

retriever: Optional[InternetRetriever] = None

if config.internet_retrieve:
    log.info(
        "Internet retriever enabled  top_k=%d  min_len=%d  timeout=%.1fs  cache=%s",
        config.ir_top_k,
        config.ir_min_len,
        config.ir_timeout,
        config.ir_cache_dir or "disabled",
    )
    retriever = InternetRetriever(
        top_k=config.ir_top_k,
        min_len=config.ir_min_len,
        timeout=config.ir_timeout,
        cache_dir=config.ir_cache_dir or None,
        use_page_fetcher=config.ir_use_page_fetcher,
    )
    # Make the retriever accessible to downstream modules via config
    config.internet_retriever = retriever
else:
    config.internet_retriever = None


# ===========================================================================
# Convenience function: retrieve_context
# ===========================================================================

def retrieve_context(question: str) -> str:
    """
    Public helper — call this from run.py / prepro.py when you need an
    internet-augmented context string for a given *question*.

    Returns an empty string when internet retrieval is disabled or fails.

    Example usage in run.py::

        from hotpot_main import retrieve_context
        extra_ctx = retrieve_context(question_text)
        # prepend extra_ctx to the model's context window
    """
    if retriever is None:
        return ""
    return retriever.format_context(question)


# ===========================================================================
# Mode dispatch  (unchanged logic from original main.py)
# ===========================================================================

if config.mode == "train":
    train(config)
elif config.mode == "prepro":
    prepro(config)
elif config.mode == "test":
    if config.internet_retrieve:
        log.info(
            "Internet retrieval is active for test mode. "
            "Import retrieve_context() from hotpot_main in run.py to use it."
        )
    test(config)
elif config.mode == "count":
    # cnt_len is not imported in the original; kept as a no-op stub
    log.warning("'count' mode referenced but cnt_len is not defined — skipping.")
else:
    log.error("Unknown mode: %s", config.mode)
