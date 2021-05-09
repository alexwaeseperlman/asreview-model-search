# Exhaustive ASReview model search
This repository allows a user to test all combinations of selected asreview model types and strategies.

## Usage
Start by running `pip install asreview[all]` to install the required packages. Then run `python test-models.py -h` to see how to run a search. It should look something like this:
```
usage: test-models.py [-h] [-o OUTPUT] [-c CLASSIFIERS [CLASSIFIERS ...]] [-q QUERY [QUERY ...]] [-b BALANCE [BALANCE ...]]
                      [-f FEATURE_EXTRACTION [FEATURE_EXTRACTION ...]] [-p PRIOR [PRIOR ...]] [-n N_INSTANCES] [-P PRESET]
                      [-s SEED]
                      filename

Test many different asreview models out on one dataset to see which perform the best

positional arguments:
  filename              Path to a labelled csv of abstracts. It should have four columns labelled: "Title", "Abstract",
                        "Authors", "Included".

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Path to a directory to output the results into. It is created if necessary
  -c CLASSIFIERS [CLASSIFIERS ...], --classifiers CLASSIFIERS [CLASSIFIERS ...]
                        List of classifiers that will be tested. The accepted options are: logistic, lstm-base, lstm-pool,
                        nb, nn-2-layer, rf, svm
  -q QUERY [QUERY ...], --query QUERY [QUERY ...]
                        List of query strategies that will be tested. The accepted options are: cluster, max, random,
                        uncertainty
  -b BALANCE [BALANCE ...], --balance BALANCE [BALANCE ...]
                        List of balancing strategies that will be tested. The accepted options are: double, simple, triple,
                        undersample
  -f FEATURE_EXTRACTION [FEATURE_EXTRACTION ...], --feature_extraction FEATURE_EXTRACTION [FEATURE_EXTRACTION ...]
                        List of feature extraction models that will be tested. The accepted options are: doc2vec, embedding-
                        idf, embedding-lstm, sbert, tfidf
  -p PRIOR [PRIOR ...], --prior PRIOR [PRIOR ...]
                        List of the number of prelabelled papers to include formatted like: prior_included,prior_excluded.
                        For example the input could look like --prior 1,1 5,5 5,10
  -n N_INSTANCES, --n_instances N_INSTANCES
                        The number of iterations per test
  -P PRESET, --preset PRESET
                        The name of the preset test to use. Valid options are: default
  -s SEED, --seed SEED  The random seed for reproducibility.
```

