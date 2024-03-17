import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data from filepath
def load_data_from_csv(filepath):
    data = pd.read_csv(filepath, sep=',', header=None, names=['asl_gloss', 'english'])
    return data

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


def main(filepath = '/home/jennamansueto/text2gloss/code/data/phoenix/phoenix.csv'):
    
    base_dir = os.path.dirname(filepath)

    print(base_dir)
    
    # Construct the paths dynamically
    split_outpath = os.path.join(base_dir, "splits/")

    data = load_data_from_csv(filepath)

    # Split data
    train, temp = train_test_split(data, test_size=0.2, random_state=42)  # 80% train, 20% for val and test
    val, test = train_test_split(temp, test_size=0.5, random_state=42)  # Split the 20% into 10% val, 10% test

    save_splits(train, split_outpath, "train")
    save_splits(val, split_outpath, "val")
    save_splits(test, split_outpath, "test")

if __name__ == "__main__":
    main()