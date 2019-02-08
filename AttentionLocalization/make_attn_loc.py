# -*- coding: utf-8 -*-

"""
Script to generate the attention localization dataset.

Running this script will save the following files in /dir/ or /dir/sample<i>/ if n_samples > 1:
- train.tsv
- validation.tsv
- test_tgt.tsv
- test_src.tsv

Help : `python make_attn_loc.py -h`
"""
import random
import csv
import os
import argparse
import sys
import warnings


### MAIN ###
def parse_arguments(args):
    desc = "Script to generate of the attention localization dataset."
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--dir', default='.',
                        help='Path to the directory where to save the generated data.')
    parser.add_argument('-s', '--n-samples', type=int, default=1,
                        help='Number of different samples to generate. If greater than 1, will save the files for a single sample in /dir/sample<i>/*.tsv.')
    parser.add_argument('--not-add-rep', action='store_true',
                        help="Don't add the repetition symbol.")
    parser.add_argument('-w', '--max-wait', type=int, default=None,
                        help="Maximum waiting token. If `None` doesn't use any.")
    parser.add_argument('-m', '--max-step', type=int, default=3,
                        help="Maximum steps that can be taken in the right or left direction.")
    parser.add_argument('-l', '--max-len-src', type=int, default=20,
                        help="Length of the source sequence is uniformly sampeled in [2 * max_step, max_len_src].")
    parser.add_argument('-T', '--max-len-tgt', type=int, default=25,
                        help="Maximum length of the target, as most targets will get stuck in some reoccuring patterns.")
    parser.add_argument('-v', '--validation-size', type=float, default=0.1,
                        help='Percentage of training set to use as validation.')
    parser.add_argument('-n', '--n-train', type=int, default=2000,
                        help='Number of examples in training set.')
    parser.add_argument('-t', '--n-test', type=int, default=500,
                        help='Number of examples in any testing set')
    parser.add_argument('--no-target-attention', action='store_true',
                        help="Don't append the target attention as an additional column.")
    parser.add_argument('--target-weights', action='store_true',
                        help="Append the target weights as an additional column.")
    parser.add_argument('-S', '--seed', type=int, default=123,
                        help='Random seed.')

    args = parser.parse_args(args)
    return args


def main(args):
    _save_arguments(vars(args), args.dir)

    random.seed(args.seed)

    for sample in range(args.n_samples):
        seed = args.seed if args.n_samples == 1 else random.randint(0, 1e5)
        out = attention_localisation_dataset(n_train=args.n_train,
                                             n_test=args.n_test,
                                             validation_size=args.validation_size,
                                             is_add_rep=not args.not_add_rep,
                                             max_wait=args.max_wait,
                                             max_step=args.max_step,
                                             max_len_src=args.max_len_src,
                                             max_len_tgt=args.max_len_tgt,
                                             is_target_attention=not args.no_target_attention,
                                             is_target_weights=args.target_weights,
                                             seed=seed)

        names = ("train", "validation", "test_src", "test_tgt")

        for data, name in zip(out, names):
            path = (args.dir if args.n_samples == 1
                    else os.path.join(args.dir, "sample{}".format(sample + 1)))
            _save_tsv(data, name, path)


### FUNCTIONS ###
def generate_src_ald(max_step=3, max_len=20, is_add_rep=True,
                     expected_n_eos=2, max_wait=None):
    """Generate a source sequence for the Attention Localisation Dataset."""
    n = random.randint(2 * (max_step) + 5, max_len)

    vocab_bounds = dict(left=dict(), right=dict())

    pos_symbols = ["+{}".format(i) for i in range(1, max_step + 1)]
    neg_symbols = [str(-i) for i in range(1, max_step + 1)]

    waiting_symbols = [] if max_wait is None else ["w{}".format(i)
                                                   for i in range(1, max_wait + 1)]

    vocab_bounds["left"]["d0"] = ["eos", "fin.", "mid."] + pos_symbols + waiting_symbols
    for i in range(1, max_step):
        v_b_l = vocab_bounds["left"]
        v_b_l["d{}".format(i)] = v_b_l["d{}".format(i - 1)] + [str(-i)]

    vocab_bounds["right"]["d0"] = ["eos", "ini.", "mid."] + neg_symbols
    vocab_bounds["right"]["d1"] = vocab_bounds["right"]["d0"] + waiting_symbols + ["+1"]
    for i in range(2, max_step):
        v_b_r = vocab_bounds["right"]
        v_b_r["d{}".format(i)] = v_b_r["d{}".format(i - 1)] + ["+{}".format(i)]

    vocab = ["eos", "fin.", "mid.", "ini."
             ] + pos_symbols + neg_symbols + waiting_symbols
    if is_add_rep:
        vocab += ["rep."]

    src = []

    weights_bounds = [0.01, 0.01, float(max_step) / 2 + 1]
    for i in range(0, max_step):
        voc = vocab_bounds["left"]["d{}".format(i)]
        src += random.choices(voc, k=1,
                              weights=weights_bounds + [1.] * (len(voc) - 3))

    # sets the correct eos weight
    middle_weights = [0., 1., 0.01, 1.] + [1.] * (len(vocab) - 4)
    eos_proba = expected_n_eos / (n - 2 * max_step)
    middle_weights = [el * (1 - eos_proba) / sum(middle_weights)
                      for el in middle_weights]
    middle_weights[0] = eos_proba
    src += random.choices(vocab, k=n - 2 * max_step, weights=middle_weights)

    for i in range(max_step - 1, -1, -1):
        voc = vocab_bounds["right"]["d{}".format(i)]
        src += random.choices(voc, k=1,
                              weights=weights_bounds + [1.] * (len(voc) - 3))

    return src


def generate_tgt_ald(src, max_len=25):
    """
    Generate the corresponding target sequence for the Attention Localisation
    Dataset.
    """
    n = len(src)
    special_idx = {"ini.": 0,
                   'mid.': (n - 1) // 2,
                   'fin.': n - 1,
                   'eos': None}

    def step(symbol, idx, last_symbol, repeat):
        if symbol == "rep.":
            symbol = last_symbol

        if symbol.startswith("w"):
            if int(symbol[1:]) > repeat:
                next_idx = idx
                repeat += 1
            else:
                repeat = 0
                next_idx = idx + 1
        elif symbol in special_idx:
            next_idx = special_idx[symbol]
        else:
            next_idx = int(symbol) + idx
        return next_idx, symbol, repeat

    tgt = []
    next_idx = 0
    length = 1
    last_symbol = None
    repeat = 0
    while length < max_len:
        tgt.append(str(next_idx))
        symbol = src[next_idx]
        next_idx, last_symbol, repeat = step(symbol, next_idx, last_symbol, repeat)
        length += 1
        if next_idx is None:
            break

    return tgt


def generate_weights_ald(src, tgt):
    """
    Generate the corresponding building block weights : ["mean_attn_old",
    "single_step", "bias"] for the Attention Localisation Dataset.
    """

    def tok_to_weights(tok, old_weights):
        try:
            weights = [1, int(tok), 0]
        except ValueError:
            if tok == "ini.":
                weights = [0, 0, -0.5]
            elif tok == "mid.":
                weights = [0, 0, 0]
            elif tok == "fin.":
                weights = [0, 0, 0.5]
            elif tok == "eos":
                weights = [1, 0, 0]
            elif tok.startswith("w"):
                weights = [1, 0, 0]
            elif tok == "rep.":
                weights = old_weights
            else:
                raise ValueError("Unkown token : {}".format(tok))

        return [str(w) for w in weights]

    old_weights = None
    weights = []
    for idx in tgt:
        tok = src[int(idx)]
        old_weights = tok_to_weights(tok, old_weights)
        weights.append(" ".join(old_weights))

    return weights


def generate_example_ald(max_step=5,
                         max_len_src=30,
                         max_len_tgt=35,
                         is_add_rep=True,
                         max_wait=None,
                         is_target_attention=False,
                         is_target_weights=False,
                         expected_n_eos=2):
    """
    Generate the corresponding src and target sequence for the Attention
    Localisation Dataset."""
    src = generate_src_ald(max_step=max_step, max_len=max_len_src,
                           is_add_rep=is_add_rep, max_wait=max_wait,
                           expected_n_eos=expected_n_eos)
    tgt = generate_tgt_ald(src, max_len=max_len_tgt)
    if is_target_weights:
        w = generate_weights_ald(src, tgt)

    out = [" ".join(src), " ".join(tgt)]

    if is_target_attention:
        out += [" ".join(tgt)]
    if is_target_weights:
        out += [" ".join(w)]
    return tuple(out)


def attention_localisation_dataset(n_train=2000,
                                   n_test=500,
                                   validation_size=0.1,
                                   is_add_rep=True,
                                   max_wait=None,
                                   max_step=3,
                                   max_len_src=20,
                                   max_len_tgt=25,
                                   expected_n_eos=2,
                                   is_target_attention=False,
                                   is_target_weights=False,
                                   seed=123):
    r"""Prepare the attention localisation dataset.

    Args:
        n_train (int, optional): number of examples in training set.
        n_test (int, optional): number of examples in any testing set. Note
            that the test set is generted such that the pattern is never the
            same as in the training set (i.e not only the src).
        validation_size (float, optional): size of the validation split.
        is_add_rep (bool, optional): whether to add the repetition symbol.
        max_wait (int, optional): maximum waiting token. If `None` doesn't use
            any. A waiting token makes it harder as the model has to count the
            number of times it attended to the same position in a row.
        max_step (int, optional): maximum steps that can be taken in the right
            or left direction.
        max_len_src (int, optional): length of the source sequence is uniformly
            sampeled in [2*max_step, max_len_src].
        max_len_tgt (int, optional): maximum length of the target, as most
            targets will get stuck in some reoccuring patterns.
        expected_n_eos (int, optional): approximate expected number of eos per
            example.
        is_target_attention (bool, optional): whether to append the target
            attention as an additional column.
        is_target_weights (bool, optional): whether to append the target
            weights as an additional column.
        seed (int, optional): sets the seed for generating random numbers.

    Return:
        train_dataset (list)
        valid_dataset (list)
        test_dataset (list)
    """
    random.seed(seed)

    n_valid = int(n_train * validation_size)
    kwargs = dict(max_step=max_step, max_len_src=max_len_src,
                  max_len_tgt=max_len_tgt, is_add_rep=is_add_rep,
                  max_wait=max_wait, is_target_attention=is_target_attention,
                  is_target_weights=is_target_weights,
                  expected_n_eos=expected_n_eos)
    train_dataset = [generate_example_ald(**kwargs) for i in range(n_train)]

    out = list(zip(*train_dataset))
    train_srcs, train_tgts = out[:2]
    train_srcs = set(train_srcs)

    if len(train_srcs) < n_train:
        warnings.warn("Only {} different train examples.".format(len(train_srcs)))

    valid_dataset = []
    while len(valid_dataset) < n_valid:
        example = generate_example_ald(**kwargs)
        if example[0] not in train_srcs:
            valid_dataset.append(example)

    test_src_dataset = []
    while len(test_src_dataset) < n_test:
        example = generate_example_ald(**kwargs)
        if example[0] not in train_srcs:
            test_src_dataset.append(example)

    test_tgt_dataset = []
    while len(test_tgt_dataset) < n_test:
        example = generate_example_ald(**kwargs)
        if example[1] not in train_tgts:
            test_tgt_dataset.append(example)

    return train_dataset, valid_dataset, test_src_dataset, test_tgt_dataset


### HELPERS ###
def _save_arguments(args, directory, filename="generation_arguments.txt"):
    """Save arguments to a file given a dictionnary."""
    with open(os.path.join(directory, filename), 'w') as file:
        file.writelines('{}={}\n'.format(k, v) for k, v in args.items())


def _save_tsv(data, name, path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

    filename = os.path.join(path, "{}.tsv".format(name))
    with open(filename, mode="w") as file:
        writer = csv.writer(file, delimiter='\t')
        for r in data:
            writer.writerow(r)


### SCRIPT ###
if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
