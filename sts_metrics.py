from nltk import word_tokenize
from nltk.translate.nist_score import sentence_nist
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk import edit_distance
from difflib import SequenceMatcher
from util import parse_sts
import argparse
import numpy as np
import sys

#### Calculate all the different metrics for TODO 2 in sts_pearson.py
# NIST
def symmetrical_nist(text_pair):
    """
    Calculates symmetrical similarity as NIST(a,b) + NIST(b,a).
    :param text_pair: iterable to two strings to compare
    :return: a float
    """

    t1,t2 = text_pair

    # Need to tokenize text to input into NIST
    t1_tokens = word_tokenize(t1.lower())
    t2_tokens = word_tokenize(t2.lower())

    # try / except to deal with ZeroDivision Error - assign 0 if error occurs
    try:
        nist_1 = sentence_nist([t1_tokens, ], t2_tokens)
    except ZeroDivisionError:
        nist_1 = 0.0

    try:
        nist_2 = sentence_nist([t2_tokens, ], t1_tokens)
    except ZeroDivisionError:
        nist_2 = 0.0

    return nist_1 + nist_2



# BLEU
def symmetrical_bleu(text_pair):
    """
    Calculates symmetrical similarity as BLEU(a,b) + BLEU(b,a).
    :param text_pair: iterable to two strings to compare
    :return: a float
    """

    t1,t2 = text_pair

    # Need to tokenize text to input into BLEU
    t1_tokens = word_tokenize(t1.lower())
    t2_tokens = word_tokenize(t2.lower())

    # try / except to deal with ZeroDivision Error - assign 0 if error occurs
    bleu_smoothing = SmoothingFunction().method4
    bleu_1 = sentence_bleu([t1_tokens, ], t2_tokens, smoothing_function = bleu_smoothing)
    bleu_2 = sentence_bleu([t2_tokens, ], t1_tokens, smoothing_function = bleu_smoothing)

    return bleu_1 + bleu_2


# LCS
def lcs_symmetrical(text_pair):
    """
    Calculates symmetrical similarity as LCS(a,b) + LCS(b,a).
    :param text_pair: iterable to two strings to compare
    :return: a float
    """

    t1,t2 = text_pair

    lcs_seq1 = SequenceMatcher(None, t1, t2, autojunk=False)
    lcs1 = lcs_seq1.find_longest_match(0, len(t1), 0, len(t2)) # returns named tuple
    lcs_size = lcs1.size

    return lcs_size


# Edit Distance
def edit_symmetrical(text_pair):
    # Automatically symmetrical - not necessary to do two different scores
    t1, t2 = text_pair
    # Calculate Levenshtein's edit distance
    ed_1 = edit_distance(t1, t2)

    return ed_1


# Word Error Rate
def wer_symmetrical(text_pair):
    # Automatically symmetrical - not necessary to do two different scores
    t1, t2 = text_pair
    t1_tokens = word_tokenize(t1.lower())
    t2_tokens = word_tokenize(t2.lower())

    # Calculate Word Error Rate
    wer = edit_distance(t1_tokens, t2_tokens)
    wer_rate = wer/(len(t1) + len(t2))

    return wer_rate
