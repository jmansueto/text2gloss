from google.cloud import storage
from tqdm import tqdm
import os
import random
import subprocess

# TO START: run 'gcloud auth application-default login' and login with your personal email

def preprocess_n_lines(bucket_name, folder_path, n, data_output_folder):
    # most preprocessing is handled by fairseq's command line instructions
    # (can use subprocess for this)
    # but we might need to do some preprocessing specific to our dataset and 
    # also create train/val/test splits
    #
    # this function should read in a certain number of lines and save the processed
    # output to the data folder
    pass

def main(bucket_name = 'cs224n-text2gloss',
    folder_path = 'data/',
    n = 2744,  # number of lines to process
    data_output_folder = "~/data/"
    ):

    data_output_folder = os.path.expanduser(data_output_folder)

    if not os.path.exists(data_output_folder):
        os.makedirs(data_output_folder)

    # Call the reading and preprocessing function
    preprocess_n_lines(bucket_name, folder_path, n, bands_to_keep, mask_band, data_output_folder)

if __name__ == "__main__":
    main()