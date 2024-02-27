import os
import random
import subprocess
import pandas as pd
# from fairseq.data import Dictionary
# from fairseq.data import FairseqDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2Tokenizer

# tokenizer = GPT2Tokenizer()

# Load data from filepath
def load_data(filepath):
    data = pd.read_csv(filepath, sep='\t', header=None, names=['english', 'asl_gloss'])
    return data

# Perform basic data cleaning
def clean_data(data):
    # Basic text cleaning
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)

    # Simple normalization
    data['english'] = data['english'].str.lower()
    data['asl_gloss'] = data['asl_gloss'].str.upper()  # Assuming ASL gloss is uppercase

# Tokenize data in one column of a pandas df
# def tokenize_data(data, column):
#     return [tokenizer.encode(sentence) for sentence in data[column]]

def save_splits(data, outpath, prefix):
    """
    Saves data language splits into separate files for English and ASL Gloss.

    Args:
    data (DataFrame): The dataset containing the splits.
    outpath (str): The directory to save the split files.
    prefix (str): The prefix for the file names (train, val, or test).
    """
    # Check if output directory exists, create if not
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    english_file_path = os.path.join(outpath, f"{prefix}.en")
    asl_file_path = os.path.join(outpath, f"{prefix}.asl")

    # Save English and ASL gloss data to separate files
    data['english'].to_csv(english_file_path, index=False, header=False)
    data['asl_gloss'].to_csv(asl_file_path, index=False, header=False)


def run_fairseq_preprocess(source_lang="en",
                           target_lang="asl",
                           trainpref="data/test_data/raw/splits/train",
                           validpref="data/test_data/raw/splits/val",
                           testpref="data/test_data/raw/splits/test",
                           destdir="data/clean",
                           workers=1):
    """
    Runs the Fairseq preprocess command with the specified parameters.

    Args:
    source_lang (str): Source language file extension.
    target_lang (str): Target language file extension.
    trainpref (str): Prefix for training data files.
    validpref (str): Prefix for validation data files.
    testpref (str): Prefix for test data files.
    destdir (str): Destination directory to store the binary files.
    workers (int): Number of worker processes to use.
    """
    if not os.path.exists(destdir):
        os.makedirs(destdir)

    command = f"fairseq-preprocess --source-lang {source_lang} --target-lang {target_lang} " \
              f"--trainpref {trainpref} --validpref {validpref} --testpref {testpref} " \
              f"--destdir {destdir} --workers {workers}"
    
    try:
        subprocess.run(command, check=True, shell=True)
        print("Fairseq preprocessing completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during Fairseq preprocessing: {e}")


def main(filepath = 'data/test_data/raw/test_data.tsv',
    split_outpath = "data/test_data/raw/splits/",
    final_outpath = "data/test_data/clean/"
    ):

    data = load_data(filepath)
    data_clean = clean_data(data)

    # Split data
    train, temp = train_test_split(data, test_size=0.2, random_state=42)  # 80% train, 20% for val and test
    val, test = train_test_split(temp, test_size=0.5, random_state=42)  # Split the 20% into 10% val, 10% test

    save_splits(train, split_outpath, "train")
    save_splits(val, split_outpath, "val")
    save_splits(test, split_outpath, "test")

    run_fairseq_preprocess(destdir=final_outpath)


if __name__ == "__main__":
    main()