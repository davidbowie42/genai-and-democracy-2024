from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from combine_datasets import create_conversation_the_conv

model_id = 'meta-llama/Meta-Llama-3-8B'

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Load datasets
dataset_the_conversation = load_dataset('csv', data_files='training_data/prepared_training_data.csv', split='train')
dataset_the_conversation = dataset_the_conversation.map(create_conversation_the_conv,
                                                        remove_columns=dataset_the_conversation.features,
                                                        batched=False)

dataset = dataset_the_conversation.train_test_split(test_size=0.2)

train_dataset = dataset['train']
eval_dataset = dataset['test']

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',            # output directory
    num_train_epochs=3,                # number of training epochs
    per_device_train_batch_size=8,     # batch size for training
    per_device_eval_batch_size=32,     # batch size for evaluation
    warmup_steps=500,                  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                 # strength of weight decay
    logging_dir='./logs',              # directory for storing logs
    logging_steps=10,
    evaluation_strategy="steps",       # evaluate every logging_steps
    save_strategy="steps",             # save model every logging_steps
    save_total_limit=2,
    # only keep the 2 most recent model checkpoints
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Start training
trainer.train()
