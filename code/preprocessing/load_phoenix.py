import zipfile
import pickle
import gzip
import csv


"""
before this run:

export KAGGLE_USERNAME=lukebabbitt
export KAGGLE_KEY=c558d67dbd4dadeffb6958b5cab774a1
kaggle datasets download -d mariusschmidtmengin/phoenixweather2014t-3rd-attempt

"""

dataset_zip_path = '/home/lukebabbitt/text2gloss/code/data/phoenix/phoenixweather2014t-3rd-attempt.zip'

# Directory to extract the dataset to
extracted_dir = '/home/lukebabbitt/text2gloss/code/data/phoenix'

with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)

# Specify the path to the CSV file
csv_file_path = "/home/lukebabbitt/text2gloss/code/data/phoenix/phoenix.csv"

# Define the header for the CSV file
header = ['gloss', 'text']

file_paths = ["/home/lukebabbitt/text2gloss/code/data/phoenix/phoenix14t.pami0.train.annotations_only.gzip", "/home/lukebabbitt/text2gloss/code/data/phoenix/phoenix14t.pami0.test.annotations_only.gzip", "/home/lukebabbitt/text2gloss/code/data/phoenix/phoenix14t.pami0.dev.annotations_only.gzip"]

with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=header)

    # Write the header
    writer.writeheader()
    for file_path in file_paths:
        with gzip.open(file_path, 'rb') as f:
            annotations = pickle.load(f)
    
            # Write data from the list of dictionaries
            for item in annotations:
                writer.writerow({'gloss': item['gloss'], 'text': item['text']})


print("CSV file has been created successfully.")