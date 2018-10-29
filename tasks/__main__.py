"""
Main entrypoint to machine-tasks

TO DO:
- update

Contact : Yann Dubois
"""
import sys
import argparse
import os

from reporter import generate_multireport
from tasks import get_task


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

DFLT_OUTPUT_DIR = os.path.join(CURRENT_DIR, "../reports/")


def rm_none_values(dictionnary):
    """Remove pairs with value `None` from a dictionnary."""
    return {k: v for k, v in dictionnary.items() if v is not None}


def str2bool(v):
    """Converts argparse string arguments to bool."""
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments(args):
    """Parse the arguments from the command line."""

    parser = argparse.ArgumentParser(description="Report Generator for the extrapolator.")

    tasks = ["lookup", "long_lookup", "long_lookup_jump", "long_lookup_oneshot",
             "long_lookup_reverse", "noisy_long_lookup_single", "noisy_long_lookup_multi",
             "long_lookup_intermediate_noise", "symbol_rewriting", "scan", "important",
             "all"]
    parser.add_argument('-t', '--tasks',
                        help='List of tasks for which to run the model. `important` and `all` are already a list of tasks. Possible: {}. (default: long lookup)'.format(tasks),
                        choices=tasks,
                        default="long lookup",
                        metavar="task",
                        nargs='+')
    parser.add_argument('-o', "--output",
                        help='Where to save the reports, absolute path. (default: ../reports/)',
                        default=None)
    # should add seed
    # should add verbosity

    # Report settings
    report = parser.add_argument_group('Report settings')
    report.add_argument('-n', '--name',
                        help='Prefix name of the report folder.',
                        default="model")
    report.add_argument('-m', '--mode',
                        help='Task mode.',
                        choices=["normal", "small", "mini"],
                        default="normal")
    report.add_argument('-k', '--k',
                        help='Number of times to rerun each task. Default given by `mode`.',
                        type=int,
                        default=None)
    report.add_argument('-c', '--compare-name',
                        help='Name of the model to which to compare to. Have to already be saved in output directory.',
                        default=None)
    report.add_argument('--no-retrain',
                        help='Whether should make the report from a saved model that should already be saved in the output directory.',
                        action='store_true',
                        default=False)

    # Train settings
    train = parser.add_argument_group('Train settings')
    train.add_argument('--teacher-forcing',
                       help='Teacher forcing ratio.',
                       default=None)
    train.add_argument('--epochs',
                       help='Maximum number of training epochs.',
                       default=None,
                       type=int)
    train.add_argument('--is-amsgrad',
                       help='Whether to use amsgrad, which should make Adam more stable.',
                       default=None,
                       type=str2bool)
    train.add_argument('--anneal-eos',
                       help='Annealing rate, increasing eos_weight.',
                       default=None,
                       type=float)
    train.add_argument('--eos-weight',
                       help='Final eos weight.',
                       default=None,
                       type=float)

    # General model settings
    model = parser.add_argument_group('General model settings')
    model.add_argument('--is-mlps',
                       help='Whether to use MLPs for the generators instead of a linear layer.',
                       default=None,
                       type=str2bool)
    model.add_argument('--is-weight-norm-rnn',
                       help='Whether to use weight normalization for the RNN.',
                       default=None,
                       type=str2bool)
    model.add_argument('--embedding-size',
                       help='Size of embedding for the decoder and encoder.',
                       default=None,
                       type=int)
    model.add_argument('--hidden-size',
                       help='Hidden size for unidirectional encoder.',
                       default=None,
                       type=int)
    model.add_argument('--anneal-mid-dropout',
                       help='Annealing rate, adding decreasing mid dropout.',
                       default=None,
                       type=float)
    model.add_argument('--anneal-mid-noise',
                       help='Annealing rate, adding decreasing mid noise.',
                       default=None,
                       type=float)
    model.add_argument('--is-transform-controller',
                       help='Whether to transform the hidden activation before using it as initialization of the decoder.',
                       default=None,
                       type=str2bool)

    # Encoder settings
    encoder = parser.add_argument_group('Encoder settings')
    encoder.add_argument('--is-res',
                         help='Whether to use a residual connection betwen the embedding and the value of the encoder.',
                         default=None,
                         type=str2bool)
    encoder.add_argument('--is-highway',
                         help='Whether to use a highway betwen the embedding and the value of the encoder.',
                         default=None,
                         type=str2bool)
    encoder.add_argument('--is-single-carry',
                         help='Whetehr to use a one dimension carry weight instead of n dimensional.',
                         default=None,
                         type=str2bool)

    # Decoder settings
    decoder = parser.add_argument_group('Decoder settings')
    decoder.add_argument('--anneal-decoder-noise-input',
                         help='Annealing rate, adding decreasing noise to the decoders input.',
                         default=None,
                         type=float)
    decoder.add_argument('--is-confuse-eos',
                         help='Remove the ability of the decoder to know the length og the longest training example.',
                         default=None,
                         type=str2bool)
    decoder.add_argument('--is-add-all-controller',
                         help="Whether to add all computed features to the decoder.",
                         default=None,
                         type=str2bool)

    # General attention settings
    attention = parser.add_argument_group('General attention settings')
    attention.add_argument('--use-attention',
                           help='Where to use attention.',
                           choices=["post-rnn", "pre-rnn", "none"],
                           default=None)
    attention.add_argument('--is-value',
                           help="Whether to use a value generator.",
                           default=None,
                           type=str2bool)
    attention.add_argument('--value-size',
                           help="Size of the generated value. -1 means same as hidden size.",
                           default=None,
                           type=int)
    attention.add_argument('--is-mid-focus',
                           help="?????????????????",
                           default=None,
                           type=str2bool)

    # Content attention settings
    content = parser.add_argument_group('Content attention settings')
    content.add_argument('--is-content-attn',
                         help="Whether to use content attention.",
                         default=None,
                         type=str2bool)
    content.add_argument('--content-method',
                         help='Content attention function to use.',
                         choices=["dot", "scaledot", "multiplicative", "additive", "hard"],
                         default=None)
    content.add_argument('--is-key',
                         help="Whether to use a key generator.",
                         default=None,
                         type=str2bool)
    content.add_argument('--key-size',
                         help="Size of the generated key. -1 means same as hidden size.",
                         default=None,
                         type=int)
    content.add_argument('--is-query',
                         help="Whether to use a query generator.",
                         default=None,
                         type=str2bool)
    content.add_argument('--anneal-kq-dropout-input',
                         help='Annealing rate, adding decreasing dropout to key query inputs.',
                         default=None,
                         type=float)
    content.add_argument('--anneal-kq-noise-input',
                         help='Annealing rate, adding decreasing noise to key query inputs.',
                         default=None,
                         type=float)
    content.add_argument('--anneal-kq-dropout-output',
                         help='Annealing rate, adding decreasing dropout to key query outputs.',
                         default=None,
                         type=float)
    content.add_argument('--anneal-kq-noise-output',
                         help='Annealing rate, adding decreasing noise to key query outputs.',
                         default=None,
                         type=float)
    content.add_argument('--is-confuse-query',
                         help='Remove the ability of the query to know what decoding step it is at.',
                         default=None,
                         type=str2bool)

    # Content attention settings
    position = parser.add_argument_group('Position attention settings')
    position.add_argument('--is-position-attn',
                          help="Whether to use positional attention.",
                          default=None,
                          type=str2bool)
    position.add_argument('--is-posrnn',
                          help="Whether to use a rnn for the positional attention.",
                          default=None,
                          type=str2bool)
    position.add_argument('--anneal-min-sigma',
                          help='Annealing rate, decreasing the minimum possible sigma.',
                          default=None,
                          type=float)
    position.add_argument('--is-bb-bias',
                          help="Adding a bias term to the positional building blocks.",
                          default=None,
                          type=str2bool)
    position.add_argument('--is-l1-bb-weights',
                          help="Whether to use a l1 regularization on the building block weights.",
                          default=None,
                          type=str2bool)
    position.add_argument('--anneal-bb-weights-noise',
                          help='Annealing rate, adding increasing noise to building block weights.',
                          default=None,
                          type=float)
    position.add_argument('--anneal-bb-noise',
                          help='Annealing rate, adding increasing noise to building block values.',
                          default=None,
                          type=float)

    args = parser.parse_args(args)

    return args


def args2tasks(args):
    """Given the parsed arguments, return teh correct list of tasks."""
    kwargs = {}
    if args.mode == "small":
        kwargs["is_small"] = True
    elif args.mode == "mini":
        kwargs["is_mini"] = True

    if args.tasks == "important":
        tasks = [get_task("long lookup", **kwargs),
                 get_task("long lookup reverse", **kwargs),
                 get_task("noisy long lookup single", **kwargs),
                 get_task("long lookup intermediate noise", **kwargs),
                 get_task("noisy long lookup multi", **kwargs),
                 get_task("scan", **kwargs),
                 get_task("symbol rewriting", is_small=True)]
    elif args.tasks == "all":
        tasks = [get_task("lookup", **kwargs),
                 get_task("long lookup", **kwargs),
                 get_task("long lookup jump", **kwargs),
                 get_task("long lookup oneshot", **kwargs),
                 get_task("long lookup reverse", **kwargs),
                 get_task("noisy long lookup single", **kwargs),
                 get_task("long lookup intermediate noise", **kwargs),
                 get_task("noisy long lookup multi", **kwargs),
                 get_task("scan", **kwargs),
                 get_task("symbol rewriting", **kwargs)]
    else:
        tasks = [get_task(task.replace("_", " "), **kwargs) for task in args.tasks]

    # will be removed after because None
    args.tasks = None
    args.mode = None
    return tasks


def main(args=None):
    """The main routine."""

    if args is None:
        args = sys.argv[1:]

    args = parse_arguments(args)

    print('Parameters: {}'.format(rm_none_values(vars(args))))
    print()

    tasks = args2tasks(args)

    output_dir = DFLT_OUTPUT_DIR if args.output is None else args.output
    args.output = None

    is_retrain = not args.no_retrain
    args.no_retrain = None

    kwargs = rm_none_values(vars(args))

    _, _ = generate_multireport(tasks,
                                output_dir,
                                is_retrain=is_retrain,
                                **kwargs)

    return 0


if __name__ == '__main__':
    sys.exit(main())
