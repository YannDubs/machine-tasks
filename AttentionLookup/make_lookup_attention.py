# -*- coding: utf-8 -*-

"""
Script to generate of the attention tables problem.

Running this script will save the following files in /dir/ or /dir/sample<i>/ if n_samples > 1:
- train.tsv
- validation.tsv
- test_add_0.tsv
...
- test_add_n.tsv

with n is `add-length-test`-1.

Help : `python make_lookup_attention.py -h`
"""
import string
import random
import csv
import os
import argparse
import sys


def parse_arguments(args):
    """Parse the given arguments."""
    def check_pair(parser, arg, name, types=(int, int)):
        if arg[0] == "None":
            arg = None
        if arg is not None and len(arg) != 2:
            raise parser.error("{} has to be None or of length 2.".format(name))
        if arg is not None:
            try:
                arg[0] = types[0](arg[0])
                arg[1] = types[1](arg[1])
            except ValueError:
                raise parser.error("{} should be of type {}".format(name, types))
        return arg

    parser = argparse.ArgumentParser(description="Script to generate of the attention lookup dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--dir', default='.', help='Path to the directory where to save the generated data.')
    parser.add_argument('-s', '--n-samples', type=int, default=5, help='Number of different samples to generate. If greater than 1, will save the files for a single sample in /dir/sample<i>/*.tsv.')

    parser.add_argument('-n', '--n-train', type=int, default=2000, help='Number of examples in training set.')
    parser.add_argument('-t', '--n-test', type=int, default=500, help='Number of examples in any testing set')
    parser.add_argument('-l', '--add-length-test', type=int, default=5, help='Additional length of the alphabet on which to test.')
    parser.add_argument('-a', '--alphabet', default=string.ascii_lowercase, help='Possible characters to use as input.')
    parser.add_argument('-A', '--alphabet-range', metavar=('MIN_ALPHABET', 'MAX_ALPHABET'), nargs='*', default=[1, 100], help='Possible range of the number of inputs.')
    parser.add_argument('-C', '--count-range', metavar=('MIN_COUNT', 'MAX_COUNT'), nargs='*', default=[1, 100], help='Possible range of the number of outputs.')
    parser.add_argument('-S', '--seed', type=int, default=123, help='Random seed.')

    args = parser.parse_args(args)

    # custom errors
    args.count_range = check_pair(parser, args.count_range, "count-range")
    args.alphabet_range = check_pair(parser, args.alphabet_range, "alphabet-range")

    return args


def main(args):
    random.seed(args.seed)
    for sample in range(args.n_samples):
        seed = args.seed if args.n_samples == 1 else random.randint(0, 1e5)
        path = args.dir if args.n_samples == 1 else os.path.join(args.dir, "sample{}".format(sample + 1))
        attention_lookup_dataset(path,
                                 n_train=args.n_train,
                                 n_test=args.n_test,
                                 add_length_test=args.add_length_test,
                                 alphabet=args.alphabet,
                                 alphabet_range=args.alphabet_range,
                                 count_range=args.count_range,
                                 seed=seed)


### FUNCTIONS ###
def attention_lookup_dataset(dir_path,
                             n_train=2000,
                             n_test=500,
                             add_length_test=5,
                             alphabet=string.ascii_lowercase,
                             alphabet_range=(1, 100),
                             count_range=(1, 100),
                             seed=123):
    """Prepare the attention lookup dataset and saves in in a `.tsv` file.

    Args:
        dir_path (str): Path to the directory where to save the generated data.
        n_train (int, optional): number of examples in training set.
        n_test (int, optional): number of examples in any testing set.
        add_length_test (int, optional): additional length of the alphabet on which to test.
        alphabet (str, optional): possible characters to use as input.
        alphabet_range (tuple, optional): possible range of the number of inputs.
        count_range (tuple, optional): possible range of the number of outputs.
        seed (int, optional): sets the seed for generating random numbers.
    """
    random.seed(seed)

    make_dir(dir_path)

    _write_attention_tsv(os.path.join(dir_path, "train.tsv"),
                         n=n_train,
                         alphabet=alphabet,
                         alphabet_range=alphabet_range,
                         count_range=count_range,
                         seed=random.randint(1, 10**20))

    _write_attention_tsv(os.path.join(dir_path, "validation.tsv"),
                         n=n_test,
                         alphabet=alphabet,
                         alphabet_range=alphabet_range,
                         count_range=count_range,
                         seed=random.randint(1, 10**20))

    for i in range(add_length_test):
        _write_attention_tsv(os.path.join(dir_path, "test_add_{}.tsv".format(i)),
                             n=n_test,
                             alphabet=alphabet,
                             alphabet_range=alphabet_range[1] + i - 1,  # forces size of alphabet
                             count_range=count_range,
                             seed=random.randint(1, 10**20),
                             min_counts=alphabet_range[1] + i - 2)


def _attention_lookup_example(alphabet=string.ascii_lowercase,
                              alphabet_range=(1, 100),
                              count_range=(1, 100),
                              seed=123,
                              min_counts=0,
                              weights_counts=None):
    """Generate and return one example of the attention lookup. Output is a list of 3 columns :
    `alphabet`, `target`, `counts`."""
    random.seed(seed)
    k_alphabet = random.choice(range(*alphabet_range)) if not isinstance(alphabet_range, int) else alphabet_range
    input_alphabet = random.choices(alphabet, k=k_alphabet)
    k_output = random.choice(range(*count_range)) if not isinstance(count_range, int) else count_range
    input_counts = random.choices(range(min_counts, k_alphabet), weights=weights_counts, k=k_output)
    target = [input_alphabet[i] for i in input_counts]

    example = [" ".join(input_alphabet), " ".join(target), " ".join(map(str, input_counts))]

    return example


def _write_attention_tsv(filename, n=1000, seed=123, **kwargs):
    """Generate `n` examples of the attention lookup and save them to a file."""
    random.seed(seed)
    with open(filename, mode="w") as file:
        writer = csv.writer(file, delimiter='\t')
        for i in range(n):
            example = _attention_lookup_example(seed=random.randint(1, 10**20), **kwargs)
            writer.writerow(example)


### HELPERS ###
def make_dir(path):
    """Makes a directory if doesn't exist."""
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


### SCRIPT ###
if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
