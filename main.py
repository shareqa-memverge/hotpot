import os
import re
import json
import string
import logging
import argparse
import unicodedata
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# Optional heavy dependencies – imported lazily so the rest of main.py works
# even when they are absent.
# ---------------------------------------------------------------------------
try:
    import requests
    from requests.adapters import HTTPAdapter, Retry
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

from prepro import prepro
from run import train, test

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ===========================================================================
# Default file-name constants (unchanged from original main.py)
# ===========================================================================

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


# ===========================================================================
# ─── Internet Retriever ────────────────────────────────────────────────────
# ===========================================================================

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _make_session(retries: int = 3, backoff: float = 0.5) -> "requests.Session":
    """Return a requests.Session with automatic retry logic."""
    if not _REQUESTS_AVAILABLE:
        raise ImportError("The 'requests' package is required for internet retrieval. "
                          "Install it with:  pip install requests")
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff,
                  status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://",  adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "HotpotQA-Retriever/1.0"})
    return session


def _normalize_text(text: str) -> str:
    """Lower-case, strip accents, collapse whitespace."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokeniser (no external deps)."""
    text = _normalize_text(text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    return [t for t in text.split() if t]


_STOPWORDS = frozenset(
    "a an the is was were are be been being have has had do does did "
    "will would shall should may might must can could of in on at to "
    "for with by from about as into through during before after above "
    "below between out off over under again further then once here "
    "there when where why how all both each few more most other some "
    "such no nor not only own same so than too very just".split()
)


def _content_tokens(text: str) -> List[str]:
    """Tokenise and remove stop-words."""
    return [t for t in _tokenize(text) if t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# TF-IDF / BM25-style scoring
# ---------------------------------------------------------------------------

def _idf(term: str, docs: List[List[str]], smoothing: float = 0.5) -> float:
    """Compute smoothed IDF for *term* over *docs* (each doc is a token list)."""
    df = sum(1 for d in docs if term in set(d))
    N  = len(docs)
    return (N - df + smoothing) / (df + smoothing)


def _bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    idf_map: Dict[str, float],
    k1: float = 1.5,
    b: float = 0.75,
    avg_dl: float = 100.0,
) -> float:
    """Return a BM25 score for one (query, document) pair."""
    dl = len(doc_tokens)
    tf_map = Counter(doc_tokens)
    score = 0.0
    for term in query_tokens:
        tf = tf_map.get(term, 0)
        if tf == 0:
            continue
        idf_val = idf_map.get(term, 0.0)
        numerator   = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / max(avg_dl, 1))
        score += idf_val * numerator / denominator
    return score


# ---------------------------------------------------------------------------
# Wikipedia search & paragraph fetching
# ---------------------------------------------------------------------------

_WIKI_SEARCH_URL  = "https://en.wikipedia.org/w/api.php"
_WIKI_PARSE_URL   = "https://en.wikipedia.org/w/api.php"
_DEFAULT_TIMEOUT  = 10   # seconds


def search_wikipedia(
    query: str,
    top_k: int = 5,
    session: Optional["requests.Session"] = None,
) -> List[Dict]:
    """
    Search Wikipedia for *query* and return up to *top_k* article titles + snippets.

    Returns
    -------
    List of dicts with keys: ``title``, ``snippet``, ``pageid``
    """
    if session is None:
        session = _make_session()

    params = {
        "action": "query",
        "list":   "search",
        "srsearch": query,
        "srlimit": top_k,
        "format": "json",
        "utf8":   1,
    }
    try:
        resp = session.get(_WIKI_SEARCH_URL, params=params, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("query", {}).get("search", [])
        return [
            {
                "title":   r["title"],
                "snippet": BeautifulSoup(r["snippet"], "html.parser").get_text()
                           if _BS4_AVAILABLE else re.sub(r"<[^>]+>", "", r["snippet"]),
                "pageid":  r["pageid"],
            }
            for r in results
        ]
    except Exception as exc:
        logger.warning("Wikipedia search failed for query %r: %s", query, exc)
        return []


def fetch_wikipedia_paragraphs(
    title: str,
    max_paragraphs: int = 10,
    session: Optional["requests.Session"] = None,
) -> List[str]:
    """
    Fetch the plain-text paragraphs of a Wikipedia article by *title*.

    Returns
    -------
    List of paragraph strings (each paragraph is a single string).
    """
    if session is None:
        session = _make_session()

    params = {
        "action":      "query",
        "prop":        "extracts",
        "exlimit":     1,
        "titles":      title,
        "explaintext": True,
        "exsectionformat": "plain",
        "format":      "json",
        "utf8":        1,
    }
    try:
        resp = session.get(_WIKI_PARSE_URL, params=params, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data  = resp.json()
        pages = data.get("query", {}).get("pages", {})
        page  = next(iter(pages.values()))
        text  = page.get("extract", "")
        if not text:
            return []

        # Split into paragraphs; keep non-trivial ones.
        raw_paras = [p.strip() for p in text.split("\n\n")]
        paragraphs = []
        for para in raw_paras:
            # Drop section headings and very short fragments
            if len(para) < 40:
                continue
            # Remove residual wiki section markers like "== Heading =="
            para = re.sub(r"^=+[^=]+=+$", "", para, flags=re.MULTILINE).strip()
            if para:
                paragraphs.append(para)
            if len(paragraphs) >= max_paragraphs:
                break
        return paragraphs
    except Exception as exc:
        logger.warning("Failed to fetch Wikipedia article %r: %s", title, exc)
        return []


def fetch_url_paragraphs(
    url: str,
    max_paragraphs: int = 10,
    session: Optional["requests.Session"] = None,
) -> Tuple[str, List[str]]:
    """
    Fetch plain-text paragraphs from an arbitrary URL.

    Returns
    -------
    (title, paragraphs)  where title is best-effort page title.
    """
    if not _BS4_AVAILABLE:
        raise ImportError("BeautifulSoup4 is required for URL fetching. "
                          "Install it with:  pip install beautifulsoup4")
    if session is None:
        session = _make_session()

    try:
        resp = session.get(url, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        soup  = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string.strip() if soup.title else url

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "form", "button"]):
            tag.decompose()

        paragraphs = []
        for p_tag in soup.find_all("p"):
            text = p_tag.get_text(" ", strip=True)
            if len(text) >= 40:
                paragraphs.append(text)
            if len(paragraphs) >= max_paragraphs:
                break
        return title, paragraphs
    except Exception as exc:
        logger.warning("Failed to fetch URL %r: %s", url, exc)
        return url, []


# ---------------------------------------------------------------------------
# Core retriever class
# ---------------------------------------------------------------------------

class InternetRetriever:
    """
    Live internet retriever for HotpotQA-style multi-hop questions.

    Workflow
    --------
    1.  Issue up to ``search_k`` Wikipedia searches derived from *question*
        (full question + each detected entity / noun phrase sub-query).
    2.  For each hit, fetch the article paragraphs from Wikipedia.
    3.  Rank all (title, paragraph) pairs with BM25 against the question.
    4.  Return the top-N pairs formatted as HotpotQA context paragraphs.

    Parameters
    ----------
    top_n_paragraphs : int
        Maximum number of paragraph-level results to return.
    search_k : int
        Number of Wikipedia article titles to fetch per sub-query.
    max_paras_per_doc : int
        How many paragraphs to read from each Wikipedia article.
    """

    def __init__(
        self,
        top_n_paragraphs: int = 10,
        search_k: int = 5,
        max_paras_per_doc: int = 8,
    ):
        self.top_n_paragraphs    = top_n_paragraphs
        self.search_k            = search_k
        self.max_paras_per_doc   = max_paras_per_doc
        self._session            = _make_session()

    # ------------------------------------------------------------------
    # Sub-query generation  (lightweight, no NLP library needed)
    # ------------------------------------------------------------------

    def _sub_queries(self, question: str) -> List[str]:
        """
        Generate a small set of search queries from *question*.

        Strategy
        --------
        - The full question itself.
        - The longest quoted phrase (if any).
        - Capitalised n-grams (likely named entities).
        """
        queries = [question]

        # Quoted spans
        for m in re.finditer(r'"([^"]{4,})"', question):
            queries.append(m.group(1))

        # Capitalized n-grams (2- and 3-word)
        words  = question.split()
        cap_tokens: List[str] = []
        for w in words:
            stripped = w.strip(string.punctuation)
            if stripped and stripped[0].isupper() and stripped.lower() not in _STOPWORDS:
                cap_tokens.append(stripped)

        if len(cap_tokens) >= 2:
            for i in range(len(cap_tokens) - 1):
                queries.append(" ".join(cap_tokens[i:i+2]))
        if len(cap_tokens) >= 3:
            for i in range(len(cap_tokens) - 2):
                queries.append(" ".join(cap_tokens[i:i+3]))

        # De-duplicate while preserving order
        seen: set = set()
        unique: List[str] = []
        for q in queries:
            key = q.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(q)
        return unique[:4]   # keep at most 4 sub-queries to limit API calls

    # ------------------------------------------------------------------
    # Main retrieval method
    # ------------------------------------------------------------------

    def retrieve(self, question: str) -> List[Dict]:
        """
        Retrieve and rank paragraphs relevant to *question*.

        Returns
        -------
        List of dicts, each with:
          ``title``     - Wikipedia article title
          ``paragraph`` - paragraph text (single string)
          ``score``     - BM25 relevance score
          ``sentences`` - list of sentence strings split from *paragraph*
        """
        logger.info("Retrieving context for: %r", question)

        sub_queries = self._sub_queries(question)
        logger.debug("Sub-queries: %s", sub_queries)

        # --- Step 1: Collect candidate (title, paragraph) pairs ----------
        title_to_paras: Dict[str, List[str]] = {}
        for sq in sub_queries:
            hits = search_wikipedia(sq, top_k=self.search_k, session=self._session)
            for hit in hits:
                title = hit["title"]
                if title in title_to_paras:
                    continue   # already fetched
                paras = fetch_wikipedia_paragraphs(
                    title,
                    max_paragraphs=self.max_paras_per_doc,
                    session=self._session,
                )
                if paras:
                    title_to_paras[title] = paras
                    logger.debug("  Fetched %d paragraphs from %r", len(paras), title)

        if not title_to_paras:
            logger.warning("No paragraphs retrieved for question: %r", question)
            return []

        # --- Step 2: Build BM25 structures --------------------------------
        query_tokens = _content_tokens(question)
        all_docs: List[Tuple[str, str, List[str]]] = []   # (title, para_text, tokens)
        for title, paras in title_to_paras.items():
            for para in paras:
                toks = _content_tokens(para)
                all_docs.append((title, para, toks))

        all_token_lists = [d[2] for d in all_docs]
        avg_dl  = sum(len(t) for t in all_token_lists) / max(len(all_token_lists), 1)
        idf_map = {term: _idf(term, all_token_lists)
                   for term in set(query_tokens)}

        # --- Step 3: Score & rank -----------------------------------------
        scored: List[Tuple[float, str, str]] = []
        for title, para, toks in all_docs:
            score = _bm25_score(query_tokens, toks, idf_map, avg_dl=avg_dl)
            scored.append((score, title, para))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:self.top_n_paragraphs]

        # --- Step 4: Format output ----------------------------------------
        results: List[Dict] = []
        for score, title, para in top:
            # Split paragraph into sentences (simple heuristic)
            sentences = _split_sentences(para)
            results.append({
                "title":     title,
                "paragraph": para,
                "score":     score,
                "sentences": sentences,
            })

        logger.info("Retrieved %d paragraphs (top score=%.3f)", len(results),
                    results[0]["score"] if results else 0.0)
        return results


# ---------------------------------------------------------------------------
# Sentence splitter (no NLTK / spaCy required)
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """
    Heuristic sentence splitter that handles common abbreviations.

    Returns a list of sentence strings.
    """
    # Protect common abbreviations
    abbrevs = r"Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|cf|al|Fig|fig|Vol|vol|No|no"
    placeholder_map: Dict[str, str] = {}
    protected = text
    for i, m in enumerate(re.finditer(r"(%s)\." % abbrevs, protected, re.IGNORECASE)):
        token = "__ABB%d__" % i
        placeholder_map[token] = m.group(0)
        protected = protected.replace(m.group(0), token, 1)

    # Split on sentence-ending punctuation followed by whitespace + capital
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", protected)

    sentences: List[str] = []
    for part in parts:
        # Restore abbreviations
        for token, original in placeholder_map.items():
            part = part.replace(token, original)
        part = part.strip()
        if part:
            sentences.append(part)
    return sentences if sentences else [text.strip()]


# ---------------------------------------------------------------------------
# HotpotQA integration helpers
# ---------------------------------------------------------------------------

def retrieve_context_for_question(
    question: str,
    top_n_paragraphs: int = 10,
    search_k: int = 5,
) -> List[List]:
    """
    High-level helper: retrieve Wikipedia paragraphs for *question* and return
    them as HotpotQA ``context`` format.

    HotpotQA context format::

        [
          [title_1, [sentence_1, sentence_2, ...]],
          [title_2, [sentence_1, sentence_2, ...]],
          ...
        ]

    Parameters
    ----------
    question : str
        The multi-hop question to retrieve evidence for.
    top_n_paragraphs : int
        How many paragraph-level hits to return in total.
    search_k : int
        Wikipedia search breadth (articles searched per sub-query).

    Returns
    -------
    List of ``[title, sentences]`` pairs ready to plug into a HotpotQA
    article's ``"context"`` field.
    """
    retriever = InternetRetriever(
        top_n_paragraphs=top_n_paragraphs,
        search_k=search_k,
    )
    raw_results = retriever.retrieve(question)

    # Convert to HotpotQA context format; merge paragraphs by title so each
    # title appears at most once (mirrors the HotpotQA data structure).
    title_to_sentences: Dict[str, List[str]] = {}
    for item in raw_results:
        title = item["title"]
        if title not in title_to_sentences:
            title_to_sentences[title] = []
        title_to_sentences[title].extend(item["sentences"])

    return [[title, sents] for title, sents in title_to_sentences.items()]


def build_hotpot_article_from_retrieved(
    question: str,
    question_id: str,
    context: List[List],
    answer: str = "",
    supporting_facts: Optional[List[List]] = None,
    question_type: str = "bridge",
) -> Dict:
    """
    Build a HotpotQA-compatible article dict from retrieved context.

    The returned dict matches the schema expected by ``prepro._process_article``
    and by the standard HotpotQA JSON data files.

    Parameters
    ----------
    question : str
        The original natural-language question.
    question_id : str
        A unique identifier string (e.g. ``"internet_retrieved_001"``).
    context : list
        List of ``[title, sentences]`` pairs (e.g. from
        ``retrieve_context_for_question``).
    answer : str
        Gold answer string if available; empty string otherwise.
    supporting_facts : list, optional
        Gold supporting-fact list ``[[title, sent_id], ...]`` if available.
    question_type : str
        ``"bridge"`` or ``"comparison"``.

    Returns
    -------
    dict following the HotpotQA article schema.
    """
    article = {
        "_id":              question_id,
        "question":         question,
        "context":          context,
        "answer":           answer,
        "supporting_facts": supporting_facts if supporting_facts is not None else [],
        "type":             question_type,
        "level":            "hard",
    }
    return article


def retrieve_and_prepro(
    question: str,
    question_id: str,
    output_json_path: str,
    answer: str = "",
    supporting_facts: Optional[List[List]] = None,
    top_n_paragraphs: int = 10,
    search_k: int = 5,
) -> str:
    """
    End-to-end pipeline: retrieve internet context for *question* and write a
    HotpotQA-compatible JSON file to *output_json_path*.

    This file can then be consumed directly by ``prepro.process_file`` or the
    ``--mode prepro`` pipeline.

    Parameters
    ----------
    question : str
        The multi-hop question.
    question_id : str
        Unique ID for this question.
    output_json_path : str
        Path where the output JSON file will be written.
    answer : str
        Gold answer if known (leave empty for inference-only use).
    supporting_facts : list, optional
        Gold supporting facts if known.
    top_n_paragraphs : int
        Retrieval breadth - number of paragraphs to fetch.
    search_k : int
        Wikipedia search breadth per sub-query.

    Returns
    -------
    str
        The path to the written JSON file (*output_json_path*).
    """
    logger.info("Running retrieve_and_prepro for question_id=%r", question_id)

    # 1. Retrieve context from the internet
    context = retrieve_context_for_question(
        question,
        top_n_paragraphs=top_n_paragraphs,
        search_k=search_k,
    )

    if not context:
        raise RuntimeError(
            "Internet retrieval returned no paragraphs for question: %r" % question
        )

    # 2. Build HotpotQA article dict
    article = build_hotpot_article_from_retrieved(
        question=question,
        question_id=question_id,
        context=context,
        answer=answer,
        supporting_facts=supporting_facts,
    )

    # 3. Wrap in a list (process_file expects a JSON array)
    data = [article]

    # 4. Write to disk
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

    logger.info("Wrote retrieved article to %r", output_json_path)
    return output_json_path


# ===========================================================================
# ─── Argument Parser (original + new internet-retriever args) ──────────────
# ===========================================================================

parser = argparse.ArgumentParser()

parser.add_argument("--mode", type=str, default="train",
                    help="One of: train | prepro | test | count | retrieve")
parser.add_argument("--data_file",       type=str)
parser.add_argument("--glove_word_file", type=str, default=glove_word_file)
parser.add_argument("--save",            type=str, default="HOTPOT")

parser.add_argument("--word_emb_file",   type=str, default=word_emb_file)
parser.add_argument("--char_emb_file",   type=str, default=char_emb_file)
parser.add_argument("--train_eval_file", type=str, default=train_eval)
parser.add_argument("--dev_eval_file",   type=str, default=dev_eval)
parser.add_argument("--test_eval_file",  type=str, default=test_eval)
parser.add_argument("--word2idx_file",   type=str, default=word2idx_file)
parser.add_argument("--char2idx_file",   type=str, default=char2idx_file)
parser.add_argument("--idx2word_file",   type=str, default=idx2word_file)
parser.add_argument("--idx2char_file",   type=str, default=idx2char_file)

parser.add_argument("--train_record_file", type=str, default=train_record_file)
parser.add_argument("--dev_record_file",   type=str, default=dev_record_file)
parser.add_argument("--test_record_file",  type=str, default=test_record_file)

parser.add_argument("--glove_char_size",  type=int, default=94)
parser.add_argument("--glove_word_size",  type=int, default=int(2.2e6))
parser.add_argument("--glove_dim",        type=int, default=300)
parser.add_argument("--char_dim",         type=int, default=8)

parser.add_argument("--para_limit",  type=int, default=1000)
parser.add_argument("--ques_limit",  type=int, default=80)
parser.add_argument("--sent_limit",  type=int, default=100)
parser.add_argument("--char_limit",  type=int, default=16)

parser.add_argument("--batch_size",  type=int,   default=64)
parser.add_argument("--checkpoint",  type=int,   default=1000)
parser.add_argument("--period",      type=int,   default=100)
parser.add_argument("--init_lr",     type=float, default=0.5)
parser.add_argument("--keep_prob",   type=float, default=0.8)
parser.add_argument("--hidden",      type=int,   default=80)
parser.add_argument("--char_hidden", type=int,   default=100)
parser.add_argument("--patience",    type=int,   default=1)
parser.add_argument("--seed",        type=int,   default=13)

parser.add_argument("--sp_lambda",       type=float, default=0.0)
parser.add_argument("--data_split",      type=str,   default="train")
parser.add_argument("--fullwiki",        action="store_true")
parser.add_argument("--prediction_file", type=str)
parser.add_argument("--sp_threshold",    type=float, default=0.3)

# ---- new args for internet retrieval mode --------------------------------
parser.add_argument(
    "--retrieve_question",
    type=str,
    default=None,
    metavar="QUESTION",
    help="(retrieve mode) The question to retrieve internet context for.",
)
parser.add_argument(
    "--retrieve_question_id",
    type=str,
    default="retrieved_q_001",
    metavar="ID",
    help="(retrieve mode) Unique question ID for the retrieved article.",
)
parser.add_argument(
    "--retrieve_output",
    type=str,
    default="retrieved_data.json",
    metavar="PATH",
    help="(retrieve mode) Output JSON file path for the retrieved article.",
)
parser.add_argument(
    "--retrieve_answer",
    type=str,
    default="",
    metavar="ANSWER",
    help="(retrieve mode) Gold answer string, if available.",
)
parser.add_argument(
    "--retrieve_top_n",
    type=int,
    default=10,
    metavar="N",
    help="(retrieve mode) Number of paragraph-level hits to retrieve.",
)
parser.add_argument(
    "--retrieve_search_k",
    type=int,
    default=5,
    metavar="K",
    help="(retrieve mode) Wikipedia search breadth per sub-query.",
)

# ===========================================================================
# ─── Config post-processing (unchanged from original) ──────────────────────
# ===========================================================================

config = parser.parse_args()


def _concat(filename: str) -> str:
    if config.fullwiki:
        return "fullwiki.{}".format(filename)
    return filename


config.dev_record_file  = _concat(config.dev_record_file)
config.test_record_file = _concat(config.test_record_file)
config.dev_eval_file    = _concat(config.dev_eval_file)
config.test_eval_file   = _concat(config.test_eval_file)

# ===========================================================================
# ─── Mode dispatch ─────────────────────────────────────────────────────────
# ===========================================================================

if config.mode == "train":
    train(config)

elif config.mode == "prepro":
    prepro(config)

elif config.mode == "test":
    test(config)

elif config.mode == "count":
    cnt_len(config)   # noqa: F821

elif config.mode == "retrieve":
    # ------------------------------------------------------------------
    # Internet retrieval mode
    # ------------------------------------------------------------------
    if not config.retrieve_question:
        parser.error("--retrieve_question is required when --mode retrieve is used.")

    output_path = retrieve_and_prepro(
        question         = config.retrieve_question,
        question_id      = config.retrieve_question_id,
        output_json_path = config.retrieve_output,
        answer           = config.retrieve_answer,
        top_n_paragraphs = config.retrieve_top_n,
        search_k         = config.retrieve_search_k,
    )
    print("Retrieved article written to:", output_path)
    print("You can now run prepro on it with:")
    print("  python main.py --mode prepro --data_file %s --data_split test" % output_path)
