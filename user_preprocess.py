# This file will be called once the setup is complete. 
# It should perform all the preprocessing steps that are necessary to run the project. 
# All important arguments will be passed via the command line.
# The input files will adhere to the format specified in datastructure/input-file.json

import json
from os.path import join, split as split_path

# TODO Implement the preprocessing steps here
def handle_input_file(file_location, output_path):
    with open(file_location) as f:
        data = json.load(f)
    
    # ...
    transformed_data = data
    transformed_data["transformed_representation"] = json.dumps([
        "If you generate tags, put your tags here (individually).",
        "If you generate embeddings, put your list of floats here.",
        "If you do all-to-en, this list should only contain one item: your translation.",
        [
            "en",
            "If you translate to multiple languages, put your translation here, prefixed by the language code."
        ],
        [
            0,
            "de",
            "You can also prepend a number to your transformed representation. This way, you can generate different representations for each sections of your input."
        ],
        [
            0,
            "Hence, there can be multiple tags or embeddings for the same section, and you can omit the language tag if it does not apply to your pipeline."
        ]
    ])
    # ...
    
    file_name = split_path(file_location)[-1]
    with open(join(output_path, file_name), "w") as f:
        json.dump(transformed_data, f)
    

# This is a useful argparse-setup, you probably want to use in your project:
import argparse
parser = argparse.ArgumentParser(description='Preprocess the data.')
parser.add_argument('--input', type=str, help='Path to the input data.', required=True, action="append")
parser.add_argument('--output', type=str, help='Path to the output directory.', required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    files_inp = args.input
    files_out = args.output
    
    for file_location in files_inp:
        handle_input_file(file_location, files_out)

 