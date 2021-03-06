import argparse
import sys
import pandas as pd
from util import parse_sts
from nltk import word_tokenize
from scipy.stats import pearsonr
from sts_metrics import symmetrical_nist, symmetrical_bleu, lcs_symmetrical, edit_symmetrical, wer_symmetrical



def main(sts_data):
    """Calculate pearson correlation between semantic similarity scores and string similarity metrics.
    Data is formatted as in the STS benchmark"""

    # TODO 1: read the dataset; implement in util.py
    texts, labels = parse_sts(sts_data)
    #print(texts)
    #print(labels)

    print(f"Found {len(texts)} STS pairs")


    # TODO 2: Calculate each of the the metrics here for each text pair in the dataset
    # HINT: Longest common substring can be complicated. Investigate difflib.SequenceMatcher for a good option.
    score_types = ["NIST", "BLEU", "Word Error Rate", "Longest common substring", "Edit Distance"]

    nist_scores = []
    bleu_scores = []
    wer_scores = []
    lcs_scores = []
    ed_scores = []

    # loop through all text pairs
    for text_pair in texts:
        # NIST
        nist_total = symmetrical_nist(text_pair)
        nist_scores.append(nist_total)
        #print(nist_scores)

        # BLEU
        bleu_total = symmetrical_bleu(text_pair)
        bleu_scores.append(bleu_total)
        #print(bleu_scores)

        # Word Error Rate
        wer_error = wer_symmetrical(text_pair)
        wer_scores.append(wer_error)
        #print(wer_scores)

        # Longest Common Substring
        lcs_ratio = lcs_symmetrical(text_pair)
        lcs_scores.append(lcs_ratio)
        #print(lcs_scores)

        # Edit Distance
        edit_total = edit_symmetrical(text_pair)
        ed_scores.append(edit_total)
        #print(ed_scores)



    #TODO 3: Calculate pearson r between each metric and the STS labels and report in the README.
    # Sample code to print results. You can alter the printing as you see fit. It is most important to put the results
    # in a table in the README

    scores_df = pd.DataFrame(
    {'NIST': nist_scores,
     'BLEU': bleu_scores,
     'WER': wer_scores,
     'LCS': lcs_scores,
     'ED': ed_scores
    })


    print(f"Semantic textual similarity for {sts_data}\n")
    for (columnName, columnData) in scores_df.iteritems():
        x = labels # pre-set labels
        y = columnData.values # calculated metrics
        score = pearsonr(x, y)
        print(f"{columnName} correlation: {score[0]:.03f}")

    # TODO 4: Complete writeup as specified by TODOs in README (describe metrics; show usage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-test.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)

