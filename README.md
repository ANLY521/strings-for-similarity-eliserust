Semantic textual similarity using string similarity
---------------------------------------------------

This project examines string similarity metrics for semantic textual similarity.
Though semantics go beyond the surface representations seen in strings, some of these
metrics constitute a good benchmark system for detecting STS.

Data is from the [STS benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark).

**TODO:**
Describe each metric in ~ 1 sentence:

BLEU = A metric that computes the amount of n-gram overlap between two sentences/phrases/words by looking at their tokens.

NIST = A variant of BLEU that also computes n-gram overlap between phrases but weights each n-gram by frequency. 

WER = Word error rate computes similarity by looking at how many "edits" (i.e. deletions, insertions, substitutions) need to occur before the two words are the same, divided by the number of words in reference. It's similar to ED but normalized.

LCS = Longest common substring computes similarity by finding the longest substring ("set of characters") shared by the text pair. For example, between "trapeze" and "trampoline" the LCS is 3: "tra".

ED = Edit distance is a variant of word error rate that looks at the shortest number of edits needed to make the text pair equivalent - using Levenshtein's distance.

**TODO:** Fill in the correlations. Expected output for DEV is provided; it is ok if your actual result
varies slightly due to preprocessing/system difference, but the difference should be quite small.

**Correlations:**

Metric | Train | Dev | Test 
------ | ----- | --- | ----
NIST | 0.496 | 0.593 | 0.475
BLEU | 0.422 | 0.433 | 0.419
WER | -0.411 | -0.452| -0.421
LCS | 0.363 | 0.468| 0.347
Edit Dist | 0.033 | -0.175| -0.039

**TODO:**
Show usage of the homework script with command line flags (see example under lab, week 1).


## lab, week 1: sts_nist.py

Calculates NIST machine translation metric for sentence pairs in an STS dataset.

Example usage:

`python sts_nist.py --sts_data stsbenchmark/sts-dev.csv`

## lab, week 2: sts_tfidf.py

Calculate pearson's correlation of semantic similarity with TFIDF vectors for text.

## homework, week 1: sts_pearson.py

Calculate pearson's correlation of semantic similarity with the metrics specified in the starter code.
Calculate the metrics between lowercased inputs and ensure that the metric is the same for either order of the 
sentences (i.e. sim(A,B) == sim(B,A)). If not, use the strategy from the lab.
Use SmoothingFunction method0 for BLEU, as described in the nltk documentation.

Run this code on the three partitions of STSBenchmark to fill in the correlations table above.
Use the --sts_data flag and edit PyCharm run configurations to run against different inputs,
 instead of altering your code for each file.