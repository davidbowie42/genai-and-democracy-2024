from test import nanoid
from user_inference import rank_articles, handle_user_query
from user_preprocess import handle_input_file
from os import listdir
from os.path import isfile, join
import json
import argparse

INPUT_ARTICLES = './sample_data'
OUTPUT_QUERIES = './output_queries'
OUTPUT_ARTICLES = './output_articles'


def query_articles(query: str):
    print(f'Query: {query}\n')

    result = handle_user_query(query, nanoid(), OUTPUT_QUERIES)

    files = [OUTPUT_ARTICLES + '/' + f for f in listdir(OUTPUT_ARTICLES) if (isfile(join(OUTPUT_ARTICLES, f)) and f.endswith('.json'))]

    articles = {}

    for file in files:
        with open(file) as f:
            d = json.load(f)

            articles[d['title']] = d['transformed_representation']

    ranked = rank_articles(result['generated_query'], articles)

    for article in ranked:
        print(article)


def preprocess():
    files = [INPUT_ARTICLES + '/' + f for f in listdir(INPUT_ARTICLES) if isfile(join(INPUT_ARTICLES, f))]

    for file in files:
        handle_input_file(file, OUTPUT_ARTICLES)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, help='The user query as string (not file path)')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess articles in sample_data folder')

    args = parser.parse_args()

    if (not args.preprocess) and (args.query is None):
        parser.print_help()

    if args.preprocess:
        preprocess()

    if args.query is not None:
        query_articles(args.query)
