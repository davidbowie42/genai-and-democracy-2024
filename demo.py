from user_inference import rank_articles, handle_user_query
from os import listdir
from os.path import isfile, join
import json

OUTPUT_QUERIES = './output_queries'
ARTICLES = './output_articles'

if __name__ == '__main__':

    query = 'le pen'

    result = handle_user_query(query, 1, OUTPUT_QUERIES)

    #print(result)

    # read all files from output_articles
    files = [ARTICLES + '/' + f for f in listdir(ARTICLES) if isfile(join(ARTICLES, f))]

    articles = {}

    for file in files:
        with open(file) as f:
            d = json.load(f)
            #print(d['transformed_representation'])
            articles[d['title']] = d['transformed_representation']

    print(rank_articles(result['generated_query'], articles))

