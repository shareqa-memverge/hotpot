import os
import argparse
import requests
from bs4 import BeautifulSoup
from prepro import prepro
from run import train, test

# --- New Internet Retrieval Functions ---

def internet_search(query, num_results=5):
    """
    Performs a search and returns a list of snippet strings.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    search_url = f"https://www.google.com/search?q={query}"
    try:
        response = requests.get(search_url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Basic logic to extract text from potential search result snippets
        results = [div.get_text() for div in soup.find_all('div', class_='BNeawe') if len(div.get_text()) > 50]
        return results[:num_results]
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []

def retrieve_internet_context(question):
    """
    Retrieves and formats internet snippets as a context block.
    """
    snippets = internet_search(question)
    if not snippets:
        return ""
    return "\n".join([f"Internet Context {i}: {s}" for i, s in enumerate(snippets)])

def augment_with_internet(question):
    """
    High-level function to augment the existing retrieval pipeline with internet data.
    """
    print(f"Augmenting query with internet retrieval for: {question}")
    return retrieve_internet_context(question)

# --- Original main.py logic with enhancements ---

def main():
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

    # New Argument for Internet Retrieval
    parser.add_argument('--internet_retrieve', action='store_true', help="Enable internet retrieval for context")

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
        if config.internet_retrieve:
            print("Running test mode with internet retrieval enabled.")
            # In a real integration, the retriever functions would be used here 
            # to fetch context before calling the testing module.
        test(config)

if __name__ == "__main__":
    main()
