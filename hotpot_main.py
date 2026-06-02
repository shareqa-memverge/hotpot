"""
hotpot_main.py
==============
Extended entry-point for the HotpotQA baseline that adds a pluggable
**Internet Retriever** layer on top of the original train/test pipeline.

Internet Retriever backends
---------------------------
  wikipedia  - query the Wikipedia MediaWiki API (no API key required)
  bing       - query the Bing Web Search v7 API (requires BING_API_KEY env-var)
  google     - query the Google Custom Search JSON API
               (requires GOOGLE_API_KEY + GOOGLE_CSE_ID env-vars)

New CLI flags
-------------
  --mode retrieve           Stand-alone retrieval mode (no training/testing)
  --retrieve                Pre-retrieve context before prepro/test modes
  --retriever {wikipedia,bing,google}
                            Which backend to use (default: wikipedia)
  --retrieve_topk INT       Max documents per question  (default: 10)
  --retrieve_input PATH     Input HotpotQA JSON file
  --retrieve_output PATH    Output path for enriched JSON
  --retrieve_replace_context
                            Replace existing context instead of appending
  --bing_api_key KEY        Bing key (overrides BING_API_KEY env-var)
  --google_api_key KEY      Google key (overrides GOOGLE_API_KEY env-var)
  --google_cse_id  ID       Google CSE ID (overrides GOOGLE_CSE_ID env-var)

Usage examples
--------------
  # Stand-alone retrieval (Wikipedia, then inspect output)
  python hotpot_main.py --mode retrieve \\
      --retrieve_input hotpot_dev_fullwiki_v1.json \\
      --retrieve_output hotpot_dev_retrieved.json \\
      --retriever wikipedia --retrieve_topk 5

  # Retrieve then preprocess (fullwiki setting)
  python hotpot_main.py --mode prepro --fullwiki \\
      --retrieve --retriever wikipedia \\
      --retrieve_input hotpot_dev_fullwiki_v1.json \\
      --retrieve_output hotpot_dev_retrieved.json \\
      --data_file hotpot_dev_retrieved.json

  # Programmatic use from another script
  from hotpot_main import retrieve_single_question
  ctx = retrieve_single_question("Who directed Inception?", retriever_name="wikipedia")
"""

import os
import re
import sys
import time
import json
import logging
import argparse
import urllib.parse
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Lazy-import pipeline modules (torch / spacy not needed for retrieval-only)
# ---------------------------------------------------------------------------
try:
    from prepro import prepro
    from run import train, test
    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Base class
# ===========================================================================

class InternetRetriever:
    """
    Abstract base class for internet-based document retrieval.

    Subclasses implement ``search(query, topk)`` and return a list of dicts::

        [
            {
                "title":    str,        # article / page title
                "url":      str,        # canonical URL
                "snippets": List[str],  # sentence-level text chunks
            },
            ...
        ]
    """

    def search(self, query: str, topk: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @staticmethod
    def results_to_context(results: List[Dict[str, Any]]) -> List[List]:
        """
        Convert retriever output to HotpotQA ``context`` format::

            [ [title, [sent0, sent1, ...]], ... ]
        """
        context = []
        for r in results:
            title    = r.get("title", "")
            snippets = r.get("snippets", [])
            if title and snippets:
                context.append([title, snippets])
        return context


# ===========================================================================
# Wikipedia retriever  (stdlib only — no third-party dependencies)
# ===========================================================================

class WikipediaRetriever(InternetRetriever):
    """
    Retrieves lead-section paragraphs from English Wikipedia via the
    public MediaWiki REST API.

    Strategy
    --------
    1. ``opensearch`` to map *query* → candidate article titles.
    2. ``extracts`` (batch) to fetch plain-text lead sections.
    3. Naive sentence-split and cap at *max_sents_per_page*.
    """

    _API      = "https://en.wikipedia.org/w/api.php"
    _TIMEOUT  = 10  # seconds
    _HEADERS  = {"User-Agent": "HotpotQA-InternetRetriever/2.0 (research)"}

    def __init__(self, max_sents_per_page: int = 15):
        self.max_sents_per_page = max_sents_per_page

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get(self, params: dict) -> dict:
        params["format"] = "json"
        url = self._API + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=self._HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=self._TIMEOUT) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, Exception) as exc:
            logger.warning("Wikipedia API error: %s", exc)
            return {}

    def _opensearch(self, query: str, limit: int) -> List[str]:
        data = self._get({
            "action":    "opensearch",
            "search":    query,
            "limit":     limit,
            "redirects": "resolve",
        })
        return data[1] if (isinstance(data, list) and len(data) >= 2) else []

    def _fetch_extracts(self, titles: List[str]) -> Dict[str, str]:
        if not titles:
            return {}
        data = self._get({
            "action":          "query",
            "prop":            "extracts",
            "exintro":         True,
            "explaintext":     True,
            "exsectionformat": "plain",
            "titles":          "|".join(titles[:20]),
            "redirects":       True,
        })
        out = {}
        for page in data.get("query", {}).get("pages", {}).values():
            t = page.get("title", "")
            e = page.get("extract", "")
            if t and e:
                out[t] = e
        return out

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Lightweight sentence splitter (no external NLP library needed)."""
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\"\'\u201c])', text)
        return [p.strip() for p in parts if p.strip()]

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def search(self, query: str, topk: int = 10) -> List[Dict[str, Any]]:
        titles   = self._opensearch(query, limit=topk)
        extracts = self._fetch_extracts(titles)
        results  = []
        for title in titles:
            text = extracts.get(title, "")
            if not text:
                continue
            sents = self._split_sentences(text)[: self.max_sents_per_page]
            results.append({
                "title":    title,
                "url":      "https://en.wikipedia.org/wiki/" +
                            urllib.parse.quote(title.replace(" ", "_")),
                "snippets": sents,
            })
        return results


# ===========================================================================
# Bing Web Search retriever
# ===========================================================================

class BingRetriever(InternetRetriever):
    """
    Retrieves snippets from the Bing Web Search v7 API.

    Requires ``BING_API_KEY`` environment variable (or *api_key* argument).
    """

    _ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"
    _TIMEOUT  = 10

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("BING_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Bing API key not set. Export BING_API_KEY or pass api_key= "
                "to BingRetriever()."
            )

    def search(self, query: str, topk: int = 10) -> List[Dict[str, Any]]:
        params = urllib.parse.urlencode({"q": query, "count": min(topk, 50)})
        url    = self._ENDPOINT + "?" + params
        req    = urllib.request.Request(
            url,
            headers={
                "Ocp-Apim-Subscription-Key": self.api_key,
                "User-Agent": "HotpotQA-InternetRetriever/2.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._TIMEOUT) as r:
                data = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, Exception) as exc:
            logger.warning("Bing API error: %s", exc)
            return []

        results = []
        for item in data.get("webPages", {}).get("value", [])[:topk]:
            raw_snippet = item.get("snippet", "")
            snippets    = [s.strip() for s in raw_snippet.split(". ") if s.strip()]
            results.append({
                "title":    item.get("name", ""),
                "url":      item.get("url", ""),
                "snippets": snippets,
            })
        return results


# ===========================================================================
# Google Custom Search retriever
# ===========================================================================

class GoogleRetriever(InternetRetriever):
    """
    Retrieves snippets from the Google Custom Search JSON API.

    Requires:
      ``GOOGLE_API_KEY`` and ``GOOGLE_CSE_ID`` environment variables
      (or *api_key* / *cse_id* constructor arguments).
    """

    _ENDPOINT = "https://www.googleapis.com/customsearch/v1"
    _TIMEOUT  = 10

    def __init__(
        self,
        api_key: Optional[str] = None,
        cse_id:  Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.cse_id  = cse_id  or os.environ.get("GOOGLE_CSE_ID",  "")
        if not self.api_key or not self.cse_id:
            raise ValueError(
                "Google credentials not set. Export GOOGLE_API_KEY and "
                "GOOGLE_CSE_ID, or pass api_key= and cse_id= to GoogleRetriever()."
            )

    def search(self, query: str, topk: int = 10) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        # Google returns max 10 per call; page through if topk > 10
        for start in range(1, topk + 1, 10):
            num    = min(10, topk - len(results))
            params = urllib.parse.urlencode({
                "key":   self.api_key,
                "cx":    self.cse_id,
                "q":     query,
                "num":   num,
                "start": start,
            })
            req = urllib.request.Request(
                self._ENDPOINT + "?" + params,
                headers={"User-Agent": "HotpotQA-InternetRetriever/2.0"},
            )
            try:
                with urllib.request.urlopen(req, timeout=self._TIMEOUT) as r:
                    data = json.loads(r.read().decode("utf-8"))
            except (urllib.error.URLError, urllib.error.HTTPError, Exception) as exc:
                logger.warning("Google CSE error: %s", exc)
                break

            items = data.get("items", [])
            if not items:
                break
            for item in items:
                snippet  = item.get("snippet", "")
                snippets = [s.strip() for s in snippet.split(". ") if s.strip()]
                results.append({
                    "title":    item.get("title", ""),
                    "url":      item.get("link", ""),
                    "snippets": snippets,
                })
            if len(results) >= topk:
                break

        return results[:topk]


# ===========================================================================
# Factory
# ===========================================================================

def build_retriever(
    name:           str,
    bing_api_key:   Optional[str] = None,
    google_api_key: Optional[str] = None,
    google_cse_id:  Optional[str] = None,
) -> InternetRetriever:
    """Instantiate and return the correct :class:`InternetRetriever` for *name*."""
    name = name.lower().strip()
    if name == "wikipedia":
        return WikipediaRetriever()
    elif name == "bing":
        return BingRetriever(api_key=bing_api_key)
    elif name == "google":
        return GoogleRetriever(api_key=google_api_key, cse_id=google_cse_id)
    raise ValueError(
        f"Unknown retriever '{name}'. Valid choices: wikipedia, bing, google"
    )


# ===========================================================================
# High-level pipeline helpers
# ===========================================================================

def retrieve_contexts_for_question(
    question:  str,
    retriever: InternetRetriever,
    topk:      int = 10,
) -> List[List]:
    """
    Run *retriever* for a single *question* string.

    Returns a HotpotQA-style ``context`` list::

        [ [title, [sent0, sent1, ...]], ... ]
    """
    results = retriever.search(question, topk=topk)
    return InternetRetriever.results_to_context(results)


def enrich_dataset_with_retrieved_context(
    input_path:             str,
    output_path:            str,
    retriever:              InternetRetriever,
    topk:                   int  = 10,
    skip_if_context_exists: bool = True,
) -> None:
    """
    Load a HotpotQA JSON file, enrich each article's ``context`` field with
    retrieved paragraphs, and save the result.

    Parameters
    ----------
    input_path             : source HotpotQA JSON (list of article dicts)
    output_path            : destination path for the enriched JSON
    retriever              : :class:`InternetRetriever` instance to use
    topk                   : max documents to retrieve per question
    skip_if_context_exists : if True, *append* retrieved docs to any existing
                             context (deduplicating by title);
                             if False, *replace* existing context entirely.
    """
    logger.info("Loading dataset: %s", input_path)
    with open(input_path, "r", encoding="utf-8") as fh:
        dataset: List[Dict[str, Any]] = json.load(fh)

    total = len(dataset)
    logger.info("Retrieving context for %d questions ...", total)

    for idx, article in enumerate(dataset):
        question = article.get("question", "").strip()
        if not question:
            continue

        retrieved = retrieve_contexts_for_question(question, retriever, topk=topk)

        if skip_if_context_exists and article.get("context"):
            existing_titles = {c[0] for c in article["context"]}
            for para in retrieved:
                if para[0] not in existing_titles:
                    article["context"].append(para)
                    existing_titles.add(para[0])
        else:
            article["context"] = retrieved

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            logger.info("  ... %d / %d questions processed", idx + 1, total)

    logger.info("Writing enriched dataset: %s", output_path)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, ensure_ascii=False, indent=2)
    logger.info("Done. %d questions written.", total)


def retrieve_single_question(
    question:       str,
    retriever_name: str           = "wikipedia",
    topk:           int           = 10,
    bing_api_key:   Optional[str] = None,
    google_api_key: Optional[str] = None,
    google_cse_id:  Optional[str] = None,
) -> List[List]:
    """
    Convenience function: retrieve HotpotQA-style context for one question.

    Example
    -------
    >>> ctx = retrieve_single_question("Who directed Inception?")
    >>> print(ctx[0][0])   # title of first retrieved article
    """
    r = build_retriever(
        retriever_name,
        bing_api_key=bing_api_key,
        google_api_key=google_api_key,
        google_cse_id=google_cse_id,
    )
    return retrieve_contexts_for_question(question, r, topk=topk)


def batch_retrieve(
    questions: List[str],
    retriever: InternetRetriever,
    topk:      int   = 10,
    delay:     float = 0.2,
) -> List[List[List]]:
    """
    Retrieve context for a list of questions.

    Parameters
    ----------
    questions : question strings
    retriever : :class:`InternetRetriever` instance
    topk      : max documents per question
    delay     : polite delay (seconds) between API calls

    Returns
    -------
    One HotpotQA-style context list per question.
    """
    contexts = []
    for i, q in enumerate(questions):
        contexts.append(retrieve_contexts_for_question(q, retriever, topk=topk))
        if delay > 0 and i < len(questions) - 1:
            time.sleep(delay)
    return contexts


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HotpotQA main script with pluggable Internet Retriever",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Original main.py arguments ----------------------------------- #
    parser.add_argument("--mode", default="train",
                        choices=["train", "prepro", "test", "count", "retrieve"],
                        help="Pipeline mode. 'retrieve' runs retrieval only.")
    parser.add_argument("--data_file",          type=str)
    parser.add_argument("--glove_word_file",    default="glove.840B.300d.txt")
    parser.add_argument("--save",               default="HOTPOT")
    parser.add_argument("--word_emb_file",      default="word_emb.json")
    parser.add_argument("--char_emb_file",      default="char_emb.json")
    parser.add_argument("--train_eval_file",    default="train_eval.json")
    parser.add_argument("--dev_eval_file",      default="dev_eval.json")
    parser.add_argument("--test_eval_file",     default="test_eval.json")
    parser.add_argument("--word2idx_file",      default="word2idx.json")
    parser.add_argument("--char2idx_file",      default="char2idx.json")
    parser.add_argument("--idx2word_file",      default="idx2word.json")
    parser.add_argument("--idx2char_file",      default="idx2char.json")
    parser.add_argument("--train_record_file",  default="train_record.pkl")
    parser.add_argument("--dev_record_file",    default="dev_record.pkl")
    parser.add_argument("--test_record_file",   default="test_record.pkl")
    parser.add_argument("--glove_char_size",    type=int,   default=94)
    parser.add_argument("--glove_word_size",    type=int,   default=int(2.2e6))
    parser.add_argument("--glove_dim",          type=int,   default=300)
    parser.add_argument("--char_dim",           type=int,   default=8)
    parser.add_argument("--para_limit",         type=int,   default=1000)
    parser.add_argument("--ques_limit",         type=int,   default=80)
    parser.add_argument("--sent_limit",         type=int,   default=100)
    parser.add_argument("--char_limit",         type=int,   default=16)
    parser.add_argument("--batch_size",         type=int,   default=64)
    parser.add_argument("--checkpoint",         type=int,   default=1000)
    parser.add_argument("--period",             type=int,   default=100)
    parser.add_argument("--init_lr",            type=float, default=0.5)
    parser.add_argument("--keep_prob",          type=float, default=0.8)
    parser.add_argument("--hidden",             type=int,   default=80)
    parser.add_argument("--char_hidden",        type=int,   default=100)
    parser.add_argument("--patience",           type=int,   default=1)
    parser.add_argument("--seed",               type=int,   default=13)
    parser.add_argument("--sp_lambda",          type=float, default=0.0)
    parser.add_argument("--data_split",         default="train")
    parser.add_argument("--fullwiki",           action="store_true")
    parser.add_argument("--prediction_file",    type=str)
    parser.add_argument("--sp_threshold",       type=float, default=0.3)

    # ---- Internet Retriever arguments --------------------------------- #
    g = parser.add_argument_group("Internet Retriever")
    g.add_argument("--retrieve", action="store_true",
                   help="Pre-retrieve context before prepro/test modes.")
    g.add_argument("--retriever", default="wikipedia",
                   choices=["wikipedia", "bing", "google"],
                   help="Retriever backend.")
    g.add_argument("--retrieve_topk", type=int, default=10,
                   help="Max documents to retrieve per question.")
    g.add_argument("--retrieve_input", type=str, default=None,
                   help="Input HotpotQA JSON file for retrieval.")
    g.add_argument("--retrieve_output", type=str, default=None,
                   help="Output path for the enriched JSON.")
    g.add_argument("--retrieve_replace_context", action="store_true",
                   help="Replace existing context instead of appending.")
    g.add_argument("--bing_api_key",   type=str, default=None,
                   help="Bing API key (overrides BING_API_KEY env-var).")
    g.add_argument("--google_api_key", type=str, default=None,
                   help="Google API key (overrides GOOGLE_API_KEY env-var).")
    g.add_argument("--google_cse_id",  type=str, default=None,
                   help="Google CSE ID (overrides GOOGLE_CSE_ID env-var).")

    return parser


# ===========================================================================
# Entry point
# ===========================================================================

def _apply_fullwiki_prefix(config) -> None:
    """Prepend 'fullwiki.' to relevant filenames when --fullwiki is active."""
    def _p(fn: str) -> str:
        return ("fullwiki." + fn) if config.fullwiki else fn

    config.dev_record_file  = _p(config.dev_record_file)
    config.test_record_file = _p(config.test_record_file)
    config.dev_eval_file    = _p(config.dev_eval_file)
    config.test_eval_file   = _p(config.test_eval_file)


def _run_retrieval(config) -> None:
    """Build retriever and call enrich_dataset_with_retrieved_context."""
    retriever = build_retriever(
        config.retriever,
        bing_api_key=config.bing_api_key,
        google_api_key=config.google_api_key,
        google_cse_id=config.google_cse_id,
    )
    enrich_dataset_with_retrieved_context(
        input_path=config.retrieve_input,
        output_path=config.retrieve_output,
        retriever=retriever,
        topk=config.retrieve_topk,
        skip_if_context_exists=not config.retrieve_replace_context,
    )


def main() -> None:
    parser = build_parser()
    config = parser.parse_args()
    _apply_fullwiki_prefix(config)

    # ------------------------------------------------------------------ #
    # Mode: retrieve (stand-alone retrieval, no model training/testing)   #
    # ------------------------------------------------------------------ #
    if config.mode == "retrieve":
        if not config.retrieve_input or not config.retrieve_output:
            parser.error(
                "--retrieve_input and --retrieve_output are required "
                "with --mode retrieve."
            )
        _run_retrieval(config)
        return

    # ------------------------------------------------------------------ #
    # Optional pre-retrieval before other pipeline modes                  #
    # ------------------------------------------------------------------ #
    if config.retrieve:
        retrieve_input  = config.retrieve_input  or config.data_file
        retrieve_output = config.retrieve_output
        if not retrieve_input:
            parser.error(
                "--retrieve requires --retrieve_input or --data_file."
            )
        if not retrieve_output:
            base, ext = os.path.splitext(retrieve_input)
            retrieve_output = base + "_retrieved" + (ext or ".json")

        config.retrieve_input  = retrieve_input
        config.retrieve_output = retrieve_output
        _run_retrieval(config)
        config.data_file = retrieve_output   # hand enriched file to prepro/test

    # ------------------------------------------------------------------ #
    # Original HotpotQA pipeline                                          #
    # ------------------------------------------------------------------ #
    if not _PIPELINE_AVAILABLE:
        logger.error(
            "Could not import prepro/run. "
            "Ensure torch, spacy and all project dependencies are installed."
        )
        sys.exit(1)

    if config.mode == "train":
        train(config)
    elif config.mode == "prepro":
        prepro(config)
    elif config.mode == "test":
        test(config)
    elif config.mode == "count":
        raise NotImplementedError(
            "'count' mode: provide a cnt_len() implementation and wire it here."
        )


if __name__ == "__main__":
    main()
