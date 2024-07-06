import json
import os

import instructor
from openai import OpenAI

from user_inference import infer_user_locale
from user_preprocess import handle_input_file, query_to_english


if __name__ == '__main__':
    input_path = './test_articles/input_articles'
    output_path = './test_articles/processed_articles'

    files = os.listdir(input_path)

    for file in files:
        file_path = os.path.join(input_path, file)

        with open(file_path) as f:
            data = json.load(f)

        content: list[str] = data["content"]
        c = "".join(content)

        client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            ),
            mode=instructor.Mode.JSON,
        )

        locale = infer_user_locale(c, client)
        print(locale)

        translated = query_to_english(c, locale, client)

        print(translated)

        #handle_input_file(file_path, output_path)
