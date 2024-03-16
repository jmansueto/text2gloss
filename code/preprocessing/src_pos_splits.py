import spacy
import os
import shutil
from tqdm import tqdm

# Load the English model
nlp = spacy.load("de_core_news_md")

def add_pos_tags(sentence):
    """Function to add POS tags to each word in a sentence."""
    doc = nlp(sentence)
    return ' '.join([f"{token.text}|{token.pos_}" for token in doc])

def process_files(input_dir, output_dir):
    """Function to process all .en files in the input directory."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir), desc="Processing files"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if filename.endswith(".en"):
            with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    modified_sentence = add_pos_tags(line.strip())
                    outfile.write(modified_sentence + '\n')

            print(f"Processed and saved to {output_path}")

        elif filename.endswith(".asl"):
            shutil.copy(input_path, output_path)
            print(f"Copied {output_path}")

# Specify the input and output directories
input_dir = '/home/jennamansueto/text2gloss/code/data/phoenix/splits'
output_dir = '/home/jennamansueto/text2gloss/code/data/phoenix_src_pos/splits'

# Process the files
process_files(input_dir, output_dir)
