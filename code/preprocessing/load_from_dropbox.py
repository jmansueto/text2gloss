import requests
from bs4 import BeautifulSoup
import re
from zipfile import ZipFile
from io import BytesIO
import sys
import csv
import os
import pandas as pd

"""
python3 code/preprocessing/load_from_dropbox.py https://achrafothman.net/site/english-asl-gloss-parallel-corpus-2012-aslg-pc12/ code/data/corpus.csv
"""


def get_dropbox_links(url):
    # make a GET request to the webpage
    response = requests.get(url)

    dropbox_links = []

    # if the request was successful
    if response.status_code == 200:
        # parse the content of the webpage
        soup = BeautifulSoup(response.text, 'html.parser')

        # find all links in the webpage
        all_links = soup.find_all('a', href=True)

        # extract dropbox links
        dropbox_links = [link['href'] for link in all_links if re.search(r'dropbox\.com', link['href'])]


    return dropbox_links


def get_txt_from_dropbox(dropbox_link):

    en_text = ""
    asl_text = ""

    if dropbox_link[-1] == "0":
        dropbox_link = dropbox_link[:-1] + "1"

    # make a GET request to the download link
    response = requests.get(dropbox_link)

    # if the request was successful
    if response.status_code == 200:
        # read the content of the response
        zip_content = BytesIO(response.content)

        # extract the content of the ZIP file
        with ZipFile(zip_content, 'r') as zip_file:
            
            # extract and print txt them
            for file_info in zip_file.infolist():
                with zip_file.open(file_info) as file:
                    print("reading " + file_info.filename)
                    content = file.read().decode('utf-8')
                    content = content.rstrip() + '\n'
                    if file_info.filename.endswith("en.txt"):
                        en_text = content
                    elif file_info.filename.endswith("asl.txt"):
                        asl_text = content

    return en_text, asl_text

def dropbox_to_csv(dropbox_links, csv_output_path):

    # open the txt files and create csv file
    with open(csv_output_path, 'a', newline='', encoding='utf-8') as csv_file:

        for dropbox_link in [dropbox_links[0]]:
            curr_en_text, curr_asl_text = get_txt_from_dropbox(dropbox_link)
            en_text = curr_en_text
            asl_text = curr_asl_text

            # split by '\n' into list of strings
            en_lines = en_text.splitlines()
            asl_lines = asl_text.splitlines()

            csv_writer = csv.writer(csv_file)

            # combine english and asl lines into CSV
            for en_line, asl_line in zip(en_lines, asl_lines):
                en_line = en_line.strip()
                asl_line = asl_line.strip()

                csv_writer.writerow([en_line, asl_line])


def main(url, csv_output_path):
    if os.path.exists(csv_output_path):
        # if it exists, delete it
        os.remove(csv_output_path)

    
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as new_csv_file:
        pass


    # get the links to dropbox located at the given url
    dropbox_links = get_dropbox_links(url)

    # convert txt at each dropbox link to csv
    dropbox_to_csv(dropbox_links, csv_output_path)



if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide URL and csv path")