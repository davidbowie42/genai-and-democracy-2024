import pandas as pd

df = pd.read_csv('training_data/the_conversation.csv')
df = df[['Body', 'Tags']]
filtered_df = df[df[['Body', 'Tags']].notnull().all(1)]

print(filtered_df.head())
print(filtered_df.info())

