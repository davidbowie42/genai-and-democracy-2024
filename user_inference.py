# This file will be executed when a user wants to query your project.
import argparse
from os.path import join
import json
import ollama
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from numpy import dot, abs
from numpy.linalg import norm


# TODO Implement the inference logic here
def handle_user_query(query, query_id, output_path):
    print(query)

    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': 'generate 3 tags from the following query:\n' + query
        }])

    tags = response['message']['content']
    print(tags)

    result = {
        "generated_queries": ["sports", "soccer", "Munich vs Dortmund"],
        "detected_language": "de",
    }

    rank_articles(["sports", "soccer", "Munich vs Dortmund"],
                  [['fc bayern', 'championship', 'munich'], ['rw essen', 'dortmund', 'football']])

    with open(join(output_path, f"{query_id}.json"), "w") as f:
        json.dump(result, f)

def infer_user_locale(query) -> str:
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': 'infer the language of the following text. return only either en, de or es. this is the text:\n' + query
        }])

    return response['message']['content']

# TODO OPTIONAL
# This function is optional for you
# You can use it to interfer with the default ranking of your system.
#
# If you do embeddings, this function will simply compute the cosine-similarity
# and return the ordering and scores
def rank_articles(generated_queries, article_representations):
    """
    This function takes as arguments the generated / augmented user query, as well as the
    transformed article representations.
    
    It needs to return a list of shape (M, 2), where M <= #article_representations.
    Each tuple contains [index, score], where index is the index in the article_repr array.
    The list need already be ordered by score. Higher is better, between 0 and 1.
    
    An empty return list indicates no matches.
    """
    query = ",".join(generated_queries)
    print(query)

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    for index, article in enumerate(article_representations):
        # Sentences we want sentence embeddings for
        tags = ",".join(article)

        sentences = [query, tags]

        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        similarity = cosine_similarity(sentence_embeddings[0], sentence_embeddings[1])

        print("Sentence embeddings:")
        print(sentence_embeddings)


def cosine_similarity(a, b):
    return abs(dot(a, b) / (norm(a) * norm(b)))


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# This is a sample argparse-setup, you probably want to use in your project:
parser = argparse.ArgumentParser(description='Run the inference.')
parser.add_argument('--query', type=str, help='The user query.', required=True, action="append")
parser.add_argument('--query_id', type=str, help='The IDs for the queries, in the same order as the queries.',
                    required=True, action="append")
parser.add_argument('--output', type=str, help='Path to the output directory.', required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    queries = args.query
    query_ids = args.query_id
    output = args.output

    print('test')

    assert len(queries) == len(query_ids), "The number of queries and query IDs must be the same."

    for query, query_id in zip(queries, query_ids):
        handle_user_query(query, query_id, output)
