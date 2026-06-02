import os
import requests
from prepro import prepro
from run import train, test
import argparse

# --- Internet Retriever Functionality ---

class InternetRetriever:
    \"\"\"
    A simple internet retriever for HotpotQA to fetch context from the web.
    Useful for the 'fullwiki' setting where external knowledge is required.
    \"\"\"
    def __init__(self, search_api_key=None):
        self.search_api_key = search_api_key

    def search_wikipedia(self, query, top_k=5):
        \"\"\"
        Search Wikipedia for the given query using the MediaWiki API.
        \"\"\"
        print(f\"[*] Searching Wikipedia for: '{query}'...\")
        search_url = \"https://en.wikipedia.org/w/api.php\"
        params = {
            \"action\": \"query\",
            \"list\": \"search\",
            \"srsearch\": query,
            \"format\": \"json\",
            \"srlimit\": top_k
        }
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            results = data.get(\"query\", {}).get(\"search\", [])
            return [res['title'] for res in results]
        except Exception as e:
            print(f\"[!] Search failed: {e}\")
            return []

    def fetch_wikipedia_content(self, title):
        \"\"\"
        Fetch the plain text content (intro) of a Wikipedia page.
        \"\"\"
        print(f\"[*] Fetching content for: '{title}'...\")
        page_url = \"https://en.wikipedia.org/w/api.php\"
        params = {
            \"action\": \"query\",
            \"prop\": \"extracts\",
            \"exintro\": True,
            \"explaintext\": True,
            \"titles\": title,
            \"format\": \"json\"
        }
        try:
            response = requests.get(page_url, params=params)
            response.raise_for_status()
            data = response.json()
            pages = data.get(\"query\", {}).get(\"pages\", {})
            for page_id in pages:
                return pages[page_id].get(\"extract\", \"\")
            return \"\"
        except Exception as e:
            print(f\"[!] Fetch failed for {title}: {e}\")
            return \"\"

    def retrieve(self, query, top_k=2):
        \"\"\"
        Retrieves relevant Wikipedia paragraphs for a given query.
        Returns a list of dictionaries with 'title' and 'text'.
        \"\"\"
        titles = self.search_wikipedia(query, top_k=top_k)
        retrieved_items = []
        for title in titles:
            text = self.fetch_wikipedia_content(title)
            if text:
                retrieved_items.append({
                    \"title\": title,
                    \"text\": text
                })
        return retrieved_items

def integrate_retriever_with_config(config):
    \"\"\"
    Example integration point: if a query is provided via CLI, 
    retrieve context and use it.
    \"\"\"
    if hasattr(config, 'query') and config.query:
        retriever = InternetRetriever()
        print(f\"\\n--- Retrieving context for query: {config.query} ---\")
        results = retriever.retrieve(config.query)
        for i, item in enumerate(results):
            print(f\"\\n[Result {i+1}] Title: {item['title']}\")
            print(f\"Content: {item['text'][:300]}...\")
        return results
    return None

# --- Main Entry Point (adapted from main.py) ---

def main():
    parser = argparse.ArgumentParser()

    glove_word_file = \"glove.840B.300d.txt\"
    word_emb_file = \"word_emb.json\"
    char_emb_file = \"char_emb.json\"
    train_eval = \"train_eval.json\"
    dev_eval = \"dev_eval.json\"
    test_eval = \"test_eval.json\"
    word2idx_file = \"word2idx.json\"
    char2idx_file = \"char2idx.json\"
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
    
    # New argument for internet retrieval
    parser.add_argument('--query', type=str, help='Query to retrieve context for from the internet')

    config = parser.parse_args()

    def _concat(filename):
        if config.fullwiki:
            return 'fullwiki.{}'.format(filename)
        return filename

    config.dev_record_file = _concat(config.dev_record_file)
    config.test_record_file = _concat(config.test_record_file)
    config.dev_eval_file = _concat(config.dev_eval_file)
    config.test_eval_file = _concat(config.test_eval_file)

    # Integration Check
    if config.query:
        integrate_retriever_with_config(config)

    if config.mode == 'train':
        train(config)
    elif config.mode == 'prepro':
        prepro(config)
    elif config.mode == 'test':
        test(config)

if __name__ == \"__main__\":
    main()
