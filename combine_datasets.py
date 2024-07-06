from datasets import load_dataset, concatenate_datasets


def create_conversation_the_conv(row):
    return {'messages': [
        {
            'role': 'user',
            'content': 'generate up to 10 tags from the following article in this format ["tag1", "tag2", "tag3", ..., "tagn"] and output nothing else:\n' +
                       row['Body']
        },
        {
            'role': 'assistant',
            'content': ''.join(row['Tags'])
        }
    ]}


def create_conversation_mongabay(row):
    return {
        'messages': [
            {
                'role': 'user',
                'content': 'generate up to 10 tags from the following article in this format ["tag1", "tag2", "tag3", ..., "tagn"] and output nothing else:\n' +
                           row['article']
            },
            {
                'role': 'assistant',
                'content': ''.join(row['tags'])
            }
        ]
    }


def create_conversation_stackexchange(row):
    return {
        'messages': [
            {
                'role': 'user',
                'content': 'generate up to 10 tags from the following query in this format ["tag1", "tag2", "tag3", ..., "tagn"] and output nothing else:\n' +
                           row['query']
            },
            {
                'role': 'assistant',
                'content': ''.join(row['tags'].split(';'))
            }
        ]
    }

def map_dataset(row):
    return { 'text': f'<s>[INST] {row["messages"][0]["content"]} [\INST] {row["messages"][1]["content"]} </s>' }

    

if __name__ == '__main__':
    dataset_the_conversation = load_dataset('csv', data_files='training_data/prepared_training_data.csv', split='train[:25%]')
    dataset_the_conversation = dataset_the_conversation.map(create_conversation_the_conv,
                                                            remove_columns=dataset_the_conversation.features,
                                                            batched=False)

    print(dataset_the_conversation.num_rows)


    print(dataset_the_conversation)

    dataset_mongabay = load_dataset('csv', data_files='training_data/mongabay.csv', split='train[:25%]')
    dataset_mongabay = dataset_mongabay.map(create_conversation_mongabay, remove_columns=dataset_mongabay.features,
                                            batched=False)

    dataset_stackexchange = load_dataset('csv', data_files='training_data/small_stackexchange.csv', split='train[:25%]')
    dataset_stackexchange = dataset_stackexchange.map(create_conversation_stackexchange, remove_columns=dataset_stackexchange.features,
                                            batched=False)
    print(dataset_stackexchange[0])

    dataset = concatenate_datasets([dataset_the_conversation, dataset_mongabay, dataset_stackexchange])
    #dataset = dataset_the_conversation
    print(dataset)
    
    dataset = dataset.map(map_dataset, remove_columns=dataset.features, batched=False)

    print(dataset.num_rows, dataset.dataset_size)

    dataset.to_json("datasets/dataset_25_percent.json", orient="records")
    print(dataset)
