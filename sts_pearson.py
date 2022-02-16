import argparse
import sys
from util import parse_sts
from nltk import word_tokenize
from scipy.stats import pearsonr
from jiwer import wer
import Levenshtein # for edit distance
from sts_metrics import symmetrical_nist, symmetrical_bleu, lcs_symmetrical



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
    text_data = zip(labels, texts)

    for label, text_pair in text_data:
        # NIST
        nist_ab = symmetrical_nist(text_pair)
        nist_ba = symmetrical_nist(text_pair)
        if nist_ab == nist_ba:
            nist_scores.append(nist_ab)

        # BLEU
        bleu_total = symmetrical_bleu(text_pair)
        bleu_scores.append(bleu_total)

        # Word Error Rate
        t1, t2 = text_pair
        wer_error = wer(t1, t2)
        wer_scores.append(wer_error)

        # Longest Common Substring
        lcs_ratio = lcs_symmetrical(text_pair)
        lcs_scores.append(lcs_ratio)

        # Edit Distance

    # Verify that all the metrics are symmetrical


    #TODO 3: Calculate pearson r between each metric and the STS labels and report in the README.
    # Sample code to print results. You can alter the printing as you see fit. It is most important to put the results
    # in a table in the README
    print(f"Semantic textual similarity for {sts_data}\n")
    for metric_name in score_types:
        score = 0.0
        print(f"{metric_name} correlation: {score:.03f}")

    # TODO 4: Complete writeup as specified by TODOs in README (describe metrics; show usage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)

