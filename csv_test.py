import pandas as pd
from datasets import load_dataset


def create_conversation(row):
    return {'messages': [
        {
            'role': 'user',
            'content': 'generate 5 tags from the following article in this format ["tag1", "tag2", "tag3", "tag4", "tag5"] and output nothing else:\n' +
                       row['Body']
        },
        {
            'role': 'assistant',
            'content': ''.join(row['Tags'])
        }
    ]}


if __name__ == '__main__':
    df = pd.read_csv('training_data/prepared_training_data.csv')

    messages = []

    dataset = load_dataset('csv', data_files='training_data/prepared_training_data.csv', split='train')

    dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)
    dataset = dataset.train_test_split(test_size=0.2)

    dataset["train"].to_json("datasets/train_dataset.json", orient="records")
    dataset["test"].to_json("datasets/test_dataset.json", orient="records")
    print(dataset)
