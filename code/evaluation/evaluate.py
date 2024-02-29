import sacrebleu
import pandas as pd

# python3 code/evaluation/evaluate.py

def compute_bleu(hypothesis, reference):
    # Compute BLEU score
    bleu_score = sacrebleu.corpus_bleu(hypotheses, [references])


    return bleu_score


if __name__ == "__main__":
    df = pd.read_csv('code/data/csv_files/corpus_0001.clean.csv', dtype=str)
    df.fillna('', inplace=True)
    print(len(df))
    hypotheses = df.iloc[0:1000, 0].tolist()
    references = df.iloc[0:1000, 1].tolist()

    print(len(references))
    print(len(hypotheses))

                  
    bleu_score = compute_bleu(hypotheses, [references])

    print('Corpus BLEU: {}'.format(bleu_score))
