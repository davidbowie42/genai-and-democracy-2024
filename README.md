# GenAI and Democracy Seminar
## Team: MORE CREATIVE

---
## Setup
- ollama needs to be running
- install all missing packages with `pip install -r requirements.txt` (should not be necessary)

## Preprocessing
Extract tags from articles and translate them to English
```bash
python3 demo.py --preprocess
```
**NOTE**: Preprocessing only needs to be executed once

## Query articles
Articles will be listed and ranked by cosine similarity
```bash
python3 demo.py --query [YOUR QUERY AS STRING (NOT A FILE)]
```

## Note
Our project uses vanilla Llama3 since finetuning did not work out as expected. Details will be in our report

