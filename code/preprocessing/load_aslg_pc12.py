import os
import pandas as pd
from datasets import load_dataset

def load_aslg_pc12():
    # Specify the dataset name from Hugging Face
    dataset_name = "aslg_pc12"

    # Load the dataset
    dataset = load_dataset(dataset_name)['train']

    # Convert the dataset to a Pandas DataFrame
    df = pd.DataFrame(dataset)

    df['gloss'] = df['gloss'].str.replace('\n', '')
    df['text'] = df['text'].str.replace('\n', '')

    # Specify the directory to save the CSV file
    output_directory = "code/data/aslg_pc12"

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Specify the CSV file name
    csv_filename = "aslg_pc12.csv"

    # Save the DataFrame to a CSV file
    csv_path = os.path.join(output_directory, csv_filename)
    df.to_csv(csv_path, index=False)


def main():
    load_aslg_pc12()

if __name__ == "__main__":
    main()