"""
Helper task objects specific for machine-tasks, that can be given to the report generator.

Contact: Yann Dubois
"""
import os

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DATA_DIR = os.path.join(CURRENT_DIR, "../")


def flatten(l):
    """Flattens a list of element or lists into a list of elements."""
    out = []
    for e in l:
        if not isinstance(e, list):
            e = [e]
        out.extend(e)
    return out


def repeat(s, n, start=1):
    """Repeats a string multiple times by adding a iter index to the name."""
    return ["{}_{}".format(s, i) for i in range(start, n + start)]


def filter_dict(d, remove):
    """Filters our the key of a dictionary."""
    return {k: v for k, v in d.items() if k not in remove}


def get_task(name,
             base_data_dir=BASE_DATA_DIR,
             is_small=False,
             is_mini=False,
             longer_repeat=5,
             loss_names=None):
    """Return the wanted tasks.

    Args:
        name ({"lookup", "long lookup", "long lookup jump", "long lookup oneshot",
            "long lookup reverse", "noisy long lookup multi", "noisy long lookup single",
            "long lookup intermediate noise", "symbol rewriting", "scan",
            "attention lookup", "attn loc", "attn loc wait", "long attn loc",
            "mini attn loc"}) name of the task to get.
        base_data_dir (str, optional): name of the base directory containing all
            the datasets.
        is_small (bool, optional): whether to run a smaller verson of the task.
            Used for getting less statistically significant results.
        is_mini (bool, optional): whether to run a smaller verson of the task.
            Used for testing purposes.
        longer_repeat (int, optional): number of longer test sets.
        loss_names (list, optional): loss_names to use. If `None` uses the task
            default one.

    Returns:
        task (tasks.tasks.Task): instantiated task.
    """
    name = name.lower()

    replace_loss_names = loss_names

    # classical lookup table
    if name == "lookup":
        task_name = "Lookup Table"
        train_file = "train"
        test_files = ["heldout_inputs", "heldout_compositions", "heldout_tables",
                      "new_compositions", "longer_compositions_seen",
                      "longer_compositions_incremental", "longer_compositions_new"]
        valid_file = "validation"
        data_dir = os.path.join(base_data_dir, "LookupTables/lookup-3bit/samples/sample1/")
        task_kwargs = {"batch_size": 1, "k": 10, "max_len": 10, "patience": 15}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = ["nll"]
        oneshot_train_file = None

    # lookup tables with training up to 3 cmpositions
    elif name == "long lookup":
        task_name = "Long Lookup Table"
        train_file = "train"
        test_files = flatten(["heldout_inputs", "heldout_compositions", "heldout_tables",
                              "new_compositions", repeat("longer_seen", longer_repeat),
                              repeat("longer_incremental", longer_repeat),
                              repeat("longer_new", longer_repeat)])
        valid_file = "validation"
        data_dir = os.path.join(base_data_dir, "LongLookupTables/sample1/")
        task_kwargs = {"batch_size": 64, "k": 3, "max_len": 15, "patience": 7}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = ["nll"]
        oneshot_train_file = None

    # lookup tables with training 1 2 and 4 compositions (i.e jumping 3)
    elif name == "long lookup jump":
        task_name = "Long Lookup Table Jump"
        train_file = "trainJump"
        test_files = flatten(["heldout_inputs", "heldout_compositions", "heldout_tables",
                              "new_compositions", repeat("longer_seen", longer_repeat),
                              repeat("longer_incremental", longer_repeat),
                              repeat("longer_new", longer_repeat)])
        valid_file = "validation"
        data_dir = os.path.join(base_data_dir, "LongLookupTables/sample1/")
        task_kwargs = {"batch_size": 64, "k": 3, "max_len": 15, "patience": 7}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = ["nll"]
        oneshot_train_file = None

    # long lookup tables with a iniital training file without t7 and t8 and then
    # adding uncomposed t7 and t8 with all the rest
    elif name == "long lookup oneshot":
        task_name = "Long Lookup Table Oneshot"
        train_file = "train_before_new_tables"
        test_files = flatten(["heldout_inputs", "heldout_compositions", "heldout_tables",
                              "new_compositions", repeat("longer_seen", longer_repeat),
                              repeat("longer_incremental", longer_repeat),
                              repeat("longer_new", longer_repeat)])
        valid_file = "validation"
        data_dir = os.path.join(base_data_dir, "LongLookupTables/sample1/")
        task_kwargs = {"batch_size": 64, "k": 3, "max_len": 15, "patience": 7}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = ["nll"]
        oneshot_train_file = "train"

    # reverse long lookup table (i.e right to left hard attention)
    elif name == "long lookup reverse":
        task_name = "Long Lookup Table Reverse"
        train_file = "train"
        test_files = flatten(["heldout_inputs", "heldout_compositions", "heldout_tables",
                              "new_compositions", repeat("longer_seen", longer_repeat),
                              repeat("longer_incremental", longer_repeat),
                              repeat("longer_new", longer_repeat)])
        valid_file = "validation"
        data_dir = os.path.join(base_data_dir, "LongLookupTablesReverse/")
        task_kwargs = {"batch_size": 64, "k": 3, "max_len": 15, "patience": 7}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = ["nll"]
        oneshot_train_file = None

    # noisy long lookup table with a special start token saying when are the
    # "real tables" starting. THe hard attention is thus a diagonal that starts at
    # some random position.
    elif name == "noisy long lookup single":
        task_name = "Noisy Long Lookup Table Single"
        train_file = "train"
        test_files = flatten(["heldout_inputs", "heldout_compositions", "heldout_tables",
                              "new_compositions", repeat("longer_seen", longer_repeat),
                              repeat("longer_incremental", longer_repeat),
                              repeat("longer_new", longer_repeat)])
        valid_file = "validation"
        data_dir = os.path.join(base_data_dir, "NoisyLongLookupTablesSingle/")
        task_kwargs = {"batch_size": 64, "k": 3, "max_len": 30, "patience": 7}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = ["nll"]
        oneshot_train_file = None

    # noisy long lookup table where there are multiple start token and only
    # the last one really counts
    elif name == "noisy long lookup multi":
        task_name = "Noisy Long Lookup Table Multi"
        train_file = "train"
        test_files = flatten(["heldout_inputs", "heldout_compositions", "heldout_tables",
                              "new_compositions", repeat("longer_seen", longer_repeat),
                              repeat("longer_incremental", longer_repeat),
                              repeat("longer_new", longer_repeat)])
        valid_file = "validation"
        data_dir = os.path.join(base_data_dir, "NoisyLongLookupTablesMulti/")
        task_kwargs = {"batch_size": 64, "k": 3, "max_len": 30, "patience": 7}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = ["nll"]
        oneshot_train_file = None

    # noisy long lookup table where between each "real" table there's one noisy
    # one. THe hard attention is thus a diagonal wich is less steep
    elif name == "long lookup intermediate noise":
        task_name = "Long Lookup Table Intermediate Noise"
        train_file = "train"
        test_files = flatten(["heldout_inputs", "heldout_compositions", "heldout_tables",
                              "new_compositions", repeat("longer_seen", longer_repeat),
                              repeat("longer_incremental", longer_repeat),
                              repeat("longer_new", longer_repeat)])
        valid_file = "validation"
        data_dir = os.path.join(base_data_dir, "LongLookupTablesIntermediateNoise/")
        task_kwargs = {"batch_size": 64, "k": 3, "max_len": 30, "patience": 7}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = ["nll"]
        oneshot_train_file = None

    # basic attention localization dataset
    elif name == "attn loc":
        task_name = "Attention Localization"
        train_file = "train"
        test_files = ["test"]
        valid_file = "validation"
        data_dir = os.path.join(base_data_dir, "AttentionLocalization")
        task_kwargs = {"batch_size": 64, "k": 3, "max_len": 25, "patience": 7,
                       "is_predict_eos": False}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = [("nll", .1), ("attention loss", 1.)]
        oneshot_train_file = None

    # basic attention localization dataset
    elif name == "attn loc wait":
        task_name = "Attention Localization Wait"
        train_file = "train"
        test_files = ["test"]
        valid_file = "validation"
        data_dir = os.path.join(base_data_dir, "AttentionLocalizationWait")
        task_kwargs = {"batch_size": 64, "k": 3, "max_len": 25, "patience": 7,
                       "is_predict_eos": False}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = [("nll", .1), ("attention loss", 1.)]
        oneshot_train_file = None

    # long attention localization dataset
    elif name == "long attn loc":
        task_name = "Long Attention Localization"
        train_file = "train"
        test_files = ["test"]
        valid_file = "validation"
        data_dir = os.path.join(base_data_dir, "LongAttentionLocalization")
        task_kwargs = {"batch_size": 64, "k": 3, "max_len": 25, "patience": 5,
                       "is_predict_eos": False}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = [("nll", .1), ("attention loss", 1.)]
        oneshot_train_file = None

    # long attention localization dataset
    elif name == "very long attn loc":
        task_name = "Very Long Attention Localization"
        train_file = "train"
        test_files = ["test"]
        valid_file = "validation"
        data_dir = os.path.join(base_data_dir, "VeryLongAttentionLocalization")
        task_kwargs = {"batch_size": 128, "k": 3, "max_len": 25, "patience": 5,
                       "is_predict_eos": False}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = [("nll", .1), ("attention loss", 1.)]
        oneshot_train_file = None

    # mini attention localization dataset
    elif name == "mini attn loc":
        task_name = "Mini Attention Localization"
        train_file = "train"
        test_files = ["test"]
        valid_file = "validation"
        data_dir = os.path.join(base_data_dir, "MiniAttentionLocalization")
        task_kwargs = {"batch_size": 64, "k": 3, "max_len": 20, "patience": 5,
                       "is_predict_eos": False}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = [("nll", .1), ("attention loss", 1.)]
        oneshot_train_file = None

    # test attention localization dataset
    elif name == "test attn loc":
        task_name = "Test Attention Localization"
        train_file = "train"
        test_files = ["test"]
        valid_file = "validation"
        data_dir = os.path.join(base_data_dir, "TestAttentionLocalization")
        task_kwargs = {"batch_size": 3, "k": 3, "max_len": 20, "patience": None,
                       "is_predict_eos": False, "eval_batch_size": 3}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = [("nll", .1), ("attention loss", 1.), ("attention mse loss", 1.)]
        oneshot_train_file = None

    # classical symbol rewriting task
    elif name == "symbol rewriting":
        task_name = "Symbol Rewriting"
        train_file = "grammar_std.train.full"
        test_files = ["grammar_long.tst.full", "grammar_repeat.tst.full",
                      "grammar_short.tst.full", "grammar_std.tst.full"]
        valid_file = "grammar.val"
        data_dir = os.path.join(base_data_dir, "SymbolRewriting/")
        task_kwargs = {"batch_size": 128, "k": 3, "max_len": 60, "patience": 5, "epochs": 20}
        metric_names = ["symbol rewriting accuracy"]
        loss_names = ["nll"]
        oneshot_train_file = None

    # classical scan
    elif name == "scan":
        task_name = "SCAN Length"
        train_file = "tasks_train_length"
        test_files = ["tasks_test_length"]
        valid_file = "tasks_validation_length"
        data_dir = os.path.join(base_data_dir, "SCAN/length_split/")
        task_kwargs = {"batch_size": 64, "k": 3, "max_len": 55, "patience": 5}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = ["nll"]
        oneshot_train_file = None

    elif name == "attention lookup":
        task_name = "Attention Lookup"
        train_file = "train"
        test_files = repeat("test_add", 5, start=0)
        valid_file = "validation"
        data_dir = os.path.join(base_data_dir, "AttentionLookup/sample1")
        task_kwargs = {"batch_size": 64, "k": 3, "max_len": 110, "patience": 5}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = ["nll"]
        oneshot_train_file = None

    else:
        raise ValueError("Unkown name : {}".format(name))

    if replace_loss_names is not None:
        loss_names = replace_loss_names

    if is_small:
        if name == "symbol rewriting":
            train_file = "grammar_std.train.small"
        task_kwargs["k"] = 1

    if is_mini:
        if name == "symbol rewriting":
            train_file = "grammar_std.train.small"
        task_kwargs["k"] = 1
        task_kwargs["batch_size"] = 128
        task_kwargs["patience"] = 2
        task_kwargs["epochs"] = 3
        task_kwargs["n_attn_plots"] = 1

    return Task(task_name, train_file, test_files, valid_file,
                data_dir=data_dir,
                task_kwargs=task_kwargs,
                metric_names=metric_names,
                loss_names=loss_names,
                oneshot_path=oneshot_train_file)


class Task(object):
    """Helper class containing meta information of datasets.

    Args:
        name (str): name of the dataset.
        train_path (str): path to training data.
        test_paths (list of str): list of paths to all test data.
        valid_path (str, optional): path to validation data.
        extension (str, optional): extension to add to every paths above.
        data_dir (str, optional):  directory to prepend to all path above.
        is_add_to_test (bool, optional): whether to add the train and validation
            path to the test paths.
        task_kwargs (dictionnaries, optional): list of task specific arguments
            that update the kwargs for a specific task.
        metric_names (list, optional): metrics to use for evaluating the task.
        metric_names (list, optional): loss to use for training the task.
        oneshot_path (str, optional): path to a file that contains the training
            examples + the new tables to use for one shot learning. If `None`
            then doesn't switch to a new training set.
    """

    def __init__(self,
                 name,
                 train_path,
                 test_paths,
                 valid_path=None,
                 extension="tsv",
                 data_dir="",
                 is_add_to_test=True,
                 task_kwargs={},
                 metric_names=["word accuracy", "sequence accuracy", "final target accuracy"],
                 loss_names=["nll"],
                 oneshot_path=None):
        self.name = name
        self.extension = "." + extension if extension != "" else extension
        self.data_dir = data_dir
        self.train_path = self._add_presufixes(train_path)
        self.test_paths = [self._add_presufixes(path) for path in test_paths]
        self.valid_path = self._add_presufixes(valid_path)
        if is_add_to_test:
            self.test_paths = [self.train_path, self.valid_path] + self.test_paths
        self.task_kwargs = task_kwargs
        self.metric_names = metric_names
        self.loss_names = loss_names
        self.oneshot_path = self._add_presufixes(oneshot_path)

    def _add_presufixes(self, path):
        if path is None:
            return None
        return os.path.join(self.data_dir, path) + self.extension
