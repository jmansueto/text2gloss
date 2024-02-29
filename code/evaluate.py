import sacrebleu
import pandas as pd

"""
python3 code/evaluate.py
"""


def compute_bleu(hypotheses, references):
    # expects hypotheses is a df of string
    # expects references is a df of df of strings (or list of just one list)

    hypotheses = hypotheses.tolist()
    for i in range(len(references)):
        references[i] = references[i].tolist()
    
    bleu_score = sacrebleu.corpus_bleu(hypotheses, references)

    print('Corpus BLEU: {}'.format(bleu_score))

    return bleu_score


def main():
    df = pd.read_csv('code/data/corpus.csv', dtype=str)
    df.fillna('', inplace=True)
    hypotheses = df.iloc[0:1000, 0]
    references = df.iloc[0:1000, 1]
                  
    bleu_score = compute_bleu(hypotheses, [references])

if __name__ == "__main__":
    main()