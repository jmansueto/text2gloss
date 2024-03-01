import sacrebleu
import pandas as pd

"""
python3 code/eval/evaluate.py
"""

REF_FILE_PATH="/home/jennamansueto/text2gloss/code/data/test_data/raw/splits/test.asl"
GEN_FILE_PATH="/home/jennamansueto/text2gloss/code/models/basic_transformer/results/generate-test.txt"


def compute_bleu(hypotheses, references):
    """
    Computes the BLEU score given hypotheses and references.

    :param hypotheses: List of generated translation strings.
    :param references: List of lists of reference translation strings.
    """
    bleu_score = sacrebleu.corpus_bleu(hypotheses, references)
    print('Corpus BLEU: {}'.format(bleu_score))
    
    return bleu_score


def read_translations(filepath):
    """
    Reads translations from a file, with one translation per line.

    :param filepath: Path to the file containing translations.
    :return: List of translations as strings.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]


def main(ref_file_path=REF_FILE_PATH, gen_file_path=GEN_FILE_PATH):
    """
    Main function to compute BLEU score from reference and generated translations files.
    """
    references = [read_translations(ref_file_path)]
    hypotheses = read_translations(gen_file_path)
    references = list(map(lambda x: [x], references[0])) # Adjusting format for sacrebleu
    bleu_score = compute_bleu(hypotheses, references)

if __name__ == "__main__":
    main()