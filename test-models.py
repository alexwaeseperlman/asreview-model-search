import argparse
import itertools
import asreview
from urllib.request import urlretrieve
import os
import zipfile
import sys
import time
import asreview.review
import asreview.analysis

presets = {
    'default': {
        'classifiers': ['logistic', 'nb', 'nn-2-layer', 'rf', 'svm'],
        'query': ['max', 'cluster'],
        'balance': ['simple'],
        'feature_extraction': ['tfidf', 'sbert', 'doc2vec'],
        'prior': ['1,1', '5,5', '10,10'],
        'n_instances': 1
    }
}

print(asreview.models.query.list_query_strategies())
parser = argparse.ArgumentParser(description='Test many different asreview models out on one dataset to see which perform the best')
parser.add_argument('filename', type=str, help='Path to a labelled csv of abstracts. It should have four columns labelled: "Title", "Abstract", "Authors", "Included".')

parser.add_argument('-o', '--output',
                    help='Path to a directory to output the results into. It is created if necessary',
                    action='store',
                    default='.',
                    type=str)

parser.add_argument('-c', '--classifiers',
                    help='List of classifiers that will be tested. The accepted options are: ' + ', '.join(asreview.models.list_classifiers()), action='extend',
                    default=asreview.models.list_classifiers(),
                    nargs='+',
                    type=str)

parser.add_argument('-q', '--query',
                    help='List of query strategies that will be tested. The accepted options are: ' + ', '.join(asreview.models.query.list_query_strategies()), 
                    action='extend',
                    default=asreview.models.query.list_query_strategies(),
                    nargs='+',
                    type=str)

parser.add_argument('-b', '--balance',
                    help='List of balancing strategies that will be tested. The accepted options are: ' + ', '.join(asreview.models.balance.list_balance_strategies()),
                    action='extend',
                    default=asreview.models.balance.list_balance_strategies(),
                    nargs='+',
                    type=str)

parser.add_argument('-f', '--feature_extraction',
                    help='List of feature extraction models that will be tested. The accepted options are: ' + ', '.join(asreview.models.feature_extraction.list_feature_extraction()),
                    action='extend',
                    default=asreview.models.feature_extraction.list_feature_extraction(),
                    nargs='+',
                    type=str)

parser.add_argument('-p', '--prior',
                    help='List of the number of prelabelled papers to include formatted like: prior_included,prior_excluded. For example the input could look like --prior 1,1 5,5 5,10',
                    action='extend',
                    default=['1,1', '5,5', '5,10'],
                    nargs='+',
                    type=str)

parser.add_argument('-n', '--n_instances',
                    help='The number of iterations per test',
                    action='store',
                    default=1,
                    type=int)

parser.add_argument('-P', '--preset',
                    help='The name of the preset test to use. Valid options are: ' + ', '.join(presets.keys()),
                    action='store',
                    default='none',
                    type=str)

parser.add_argument('-s', '--seed',
                    help='The random seed for reproducibility.',
                    action='store',
                    default=0,
                    type=int)

args = parser.parse_args()
print(args)

if args.preset in presets:
    for k, v in presets[args.preset].items():
        args.__dict__[k] = v

# TODO: Download embedding files and enable lstm models


if not os.path.exists(args.filename):
    raise FileNotFoundError('Data file \'' + args.filename + '\' not found.')

print()
print(f'Model testing will take {len(args.classifiers) * len(args.query) * len(args.balance) * len(args.feature_extraction) * len(args.prior) * args.n_instances} iterations.\n'
    + 'Each iteration can take up to an hour, and the output files can be up to a few GB, depending on the type of model and dataset used.\n'
    + 'Are you sure you would like to continue? [Yn]', end=' ')

resp = input()

if resp.lower().startswith('n'):
    print('Exiting.')
    sys.exit(0)

os.makedirs(args.output, exist_ok=True)

print('Running models')

dataset = asreview.ASReviewData.from_file(args.filename)

# Try a simulation with every combination of the inputted options
for classifier_name, query_name, balance_name, feature_extraction_name, prior in itertools.product(args.classifiers, args.query, args.balance, args.feature_extraction, args.prior):

    args.seed += 1

    print(f"Classifier: '{classifier_name}', feature extraction: '{feature_extraction_name}', query strategy: '{query_name}', balancing strategy: '{balance_name}', prior amounts: '{prior}'")
    start_time = time.time()
    # TODO: Enable lstm models
    if 'lstm' in classifier_name or 'lstm' in feature_extraction_name:
        print('Skipping iteration because lstm models are not supported')

    classifier = asreview.models.classifiers.get_classifier(classifier_name)
    query = asreview.models.query.get_query_model(query_name)
    balance = asreview.models.balance.get_balance_model(balance_name)
    feature_extraction = asreview.models.feature_extraction.get_feature_model(feature_extraction_name)
    prior_included, prior_excluded = [int(i) for i in prior.split(',')]

    asreview.ReviewSimulate(dataset,
                        model=classifier,
                        query_model=query,
                        balance_model=balance,
                        feature_model=feature_extraction,
                        n_prior_included=prior_included,
                        n_prior_excluded=prior_excluded,
                        state_file=os.path.join(args.output,
                                f'{classifier_name}-{query_name}-{balance_name}-{feature_extraction_name}-{prior}.h5')
                        ).review()


    end_time = time.time()
    print('Finished in', (end_time - start_time), 'seconds')





