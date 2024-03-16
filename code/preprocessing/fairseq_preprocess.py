import os
import random
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

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

def main(task, tokenizer, source_lang, workers, target_lang, bpe_type, filepath = '/home/jennamansueto/text2gloss/code/data/phoenix_src_pos/phoenix.csv',
    ):
    
    base_dir = os.path.dirname(filepath)
    
    # Construct the paths dynamically
    split_outpath = os.path.join(base_dir, "bpe_splits/")
    final_outpath = os.path.join(base_dir, "clean/")


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