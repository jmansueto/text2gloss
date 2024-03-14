import os
import random
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

# Load data from filepath
def load_data_from_csv(filepath):
    data = pd.read_csv(filepath, sep=',', header=None, names=['asl_gloss', 'english'])
    return data

# generate bpe codes
def generate_bpe_codes(input_file, output_codes_file, num_operations):
    """
    Generates BPE codes for the given input file.

    Args:
    input_file (str): Path to the text file for generating BPE codes.
    output_codes_file (str): Path where the BPE codes will be saved.
    num_operations (int): Number of BPE merge operations.
    """
    command = f"subword-nmt learn-bpe -s {num_operations} < {input_file} > {output_codes_file}"
    os.system(command)

def apply_bpe_to_file(input_file, output_file, bpe_codes_file):
    """
    Applies BPE encoding to a file using the specified BPE codes.

    Args:
    input_file (str): Path to the input text file.
    output_file (str): Path where the BPE-encoded text will be saved.
    bpe_codes_file (str): Path to the BPE codes file.
    """
    command = f"subword-nmt apply-bpe -c {bpe_codes_file} < {input_file} > {output_file}"
    os.system(command)


# Perform basic data cleaning
def clean_data(data):
    # Basic text cleaning
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)

    # Simple normalization
    data['english'] = data['english'].str.lower()
    data['asl_gloss'] = data['asl_gloss'].str.upper()  # Assuming ASL gloss is uppercase


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


def run_fairseq_preprocess(pref,
                        destdir, task,
                        source_lang,
                        target_lang,
                        workers,
                        tokenizer,
                        bpe_type                    
    ):
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
    trainpref = os.path.join(pref, 'train')
    validpref = os.path.join(pref, 'val')
    testpref = os.path.join(pref, 'test')

    if not os.path.exists(destdir):
        os.makedirs(destdir)


    command = f"fairseq-preprocess --source-lang {source_lang} --target-lang {target_lang} " \
              f"--trainpref {trainpref} --validpref {validpref} --testpref {testpref} --bpe {bpe_type} " \
              f"--destdir {destdir} --workers {workers} --tokenizer {tokenizer} --task {task}"

    
    try:
        subprocess.run(command, check=True, shell=True)
        print("Fairseq preprocessing completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during Fairseq preprocessing: {e}")


"""
Automatically gennerates the following file structure:
--> TEST_DATA (or whatever your folder is called)
-----> clean
----------> {output of fairseq preprocess, i.e. tokenized binarized split files}
-----> splits
----------> train val test splits (clean but un-tokenized/binarized)
-----> bpe_codes.txt
"""

def main(task, tokenizer, source_lang, workers, target_lang, bpe_type, filepath = 'code/data/phoenix/phoenix.csv',
    num_merge_ops=32000
    ):
    
    base_dir = os.path.dirname(filepath)

    print(base_dir)
    
    # Construct the paths dynamically
    split_outpath = os.path.join(base_dir, "splits/")
    final_outpath = os.path.join(base_dir, "clean/")
    bpe_codes_path = base_dir + "/bpe_codes.txt"
    

    data = load_data_from_csv(filepath)
    data_clean = clean_data(data)

    # Split data
    train, temp = train_test_split(data, test_size=0.2, random_state=42)  # 80% train, 20% for val and test
    val, test = train_test_split(temp, test_size=0.5, random_state=42)  # Split the 20% into 10% val, 10% test

    save_splits(train, split_outpath, "train")
    save_splits(val, split_outpath, "val")
    save_splits(test, split_outpath, "test")

    # Generate BPE codes (based on the training data)
    generate_bpe_codes(os.path.join(split_outpath, "train.en"), bpe_codes_path, num_merge_ops)

    run_fairseq_preprocess(destdir=final_outpath, 
                            pref=split_outpath, 
                            task=task, 
                            tokenizer=tokenizer, 
                            source_lang=source_lang,
                            target_lang=target_lang,
                            workers=workers,
                            bpe_type=bpe_type
                            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data and run Fairseq preprocess.")
    parser.add_argument("--task", default="translation", help="Name of the model.")
    parser.add_argument("--tokenizer", default="moses", help="Tokenizer.")
    parser.add_argument("--source_lang", default="en", help="source_lang.")
    parser.add_argument("--target_lang", default="asl", help="target_lang.")
    parser.add_argument("--workers", default=1, help="workers.")
    parser.add_argument("--bpe_type", default="subword_nmt", help="bpe_type.") 
    
    args = parser.parse_args()
    
    main(task=args.task, 
        tokenizer=args.tokenizer, 
        source_lang=args.source_lang, 
        target_lang=args.target_lang, 
        workers=args.workers, 
        bpe_type=args.bpe_type
        )