# coding: utf-8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import pearsonr
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from util import parse_sts
import argparse
import sys
import warnings

warnings.filterwarnings("ignore")


def preprocess_text(text):
    """Preprocess one sentence:
    tokenizes, lowercases, applies the Porter stemmer,
     removes punctuation tokens and stopwords.
     Returns a string of tokens joined by whitespace.
     """
    # Stemmer
    stemmer = PorterStemmer()
    stops = set(stopwords.words('english')) # Identify  stop words
    toks = word_tokenize(text) # Tokenize sentence
    toks_stemmed = [stemmer.stem(tok.lower()) for tok in toks] # STEM tokens
    toks_nopunc = [tok for tok in toks_stemmed if tok not in string.punctuation] # Remove punctuation
    toks_nostop = [tok for tok in toks_nopunc if tok not in stops] # Remove stopwords
    return " ".join(toks_nostop)


def main(sts_data):

    texts, labels = parse_sts(sts_data)

    # TODO 1: get a single list of texts to determine vocabulary and document frequency

    # get a single list of texts to determine vocabulary and document frequency --> input one list into TfidfVectorizer
    all_t1, all_t2 = zip(*texts)
    print(f"{len(all_t1)} texts total")
    all_texts = all_t1 + all_t2

    # create a TfidfVectorizer
    # input = content bc dealing with list of strings
    vectorizer = TfidfVectorizer(input = "content", lowercase = True, analyzer = "word", use_idf = True, min_df = 10)

    # fit to the training data
    tfidf_vector = vectorizer.fit(all_texts)
    print("Checking the vocabulary: ")
    print(tfidf_vector.get_feature_names())


    # TODO 2: Can normalization like removing stopwords remove differences that aren't meaningful?
    # define the function preprocess_text above
    preproc_train_texts = [preprocess_text(text) for text in texts]
    print(preproc_train_texts)


    # TODO 3: Learn another TfidfVectorizer for preprocessed data
    # Use token_pattern "\S+" in the TfidfVectorizer to split on spaces
    vectorizer_preproc = TfidfVectorizer(input = "content", lowercase = True, analyzer = "word", use_idf = True,
                                         min_df = 10, token_pattern = "\S+")
    tfidf_preproc = vectorizer_preproc.fit(preproc_train_texts)
    print(tfidf_preproc.get_feature_names())

    # TODO 4: compute cosine similarity for each pair of sentences, both with and without preprocessing
    cos_sims = []
    cos_sims_preproc = []

    for pair in texts:
        t1,t2 = pair
        pair_reprs = vectorizer.transform([t1, t2])
        pair_similarity = cosine_similarity(pair_reprs[0], pair_reprs[1])
        print(pair_similarity)
        cos_sims.append(pair_similarity[0,0])

        t1_preproc = preprocess_text(t1)
        t2_preproc = preprocess_text(t2)
        pair_reprs = vectorizer_preproc([t1_preproc, t2_preproc])
        pair_similarity = cosine_similarity(pair_reprs[0], pair_reprs[1])
        print(pair_similarity)
        cos_sims_preproc.append(pair_similarity[0,0])

    # TODO 5: measure the correlations
    pearson = pearsonr(cos_sims, labels)
    preproc_pearson = pearsonr(cos_sims_preproc, labels)
    print(f"default settings: r={pearson:.03}")
    print(f"preprocessed text: r={preproc_pearson:.03}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)