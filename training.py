from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

model_id = 'meta-llama/Meta-Llama-3-8B'

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Load datasets
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
train_dataset = dataset['train']
eval_dataset = dataset['validation']

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

trainer.save