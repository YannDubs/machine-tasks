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
    return ["{}_{}".format(s, i) for i in range(start, n + start)]


def filter_dict(d, remove):
    return {k: v for k, v in d.items() if k not in remove}


def get_task(name, base_data_dir=BASE_DATA_DIR, is_small=False, is_mini=False, longer_repeat=5, **kwargs):
    name = name.lower()

    if name == "lookup":
        task_name = "Lookup Table"
        train_file = "train"
        test_files = ["heldout_inputs", "heldout_compositions", "heldout_tables", "new_compositions",
                      "longer_compositions_seen", "longer_compositions_incremental", "longer_compositions_new"]
        valid_file = "validation"
        data_dir = os.path.join(base_data_dir, "LookupTables/lookup-3bit/samples/sample1/")
        task_kwargs = {"batch_size": 1, "k": 10, "max_len": 10, "patience": 15}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = ["nll"]

    elif name == "long lookup":
        task_name = "Long Lookup Table"
        train_file = "train"
        test_files = flatten(["heldout_inputs", "heldout_compositions", "heldout_tables", "new_compositions",
                              repeat("longer_seen", longer_repeat), repeat("longer_incremental", longer_repeat),
                              repeat("longer_new", longer_repeat)])
        valid_file = "validation"
        data_dir = os.path.join(base_data_dir, "LongLookupTables/sample1/")
        task_kwargs = {"batch_size": 64, "k": 3, "max_len": 15, "patience": 5}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = ["nll"]

    elif name == "guided long lookup":
        task_name = "Long Lookup Table"
        train_file = "train"
        test_files = flatten(["heldout_inputs", "heldout_compositions", "heldout_tables", "new_compositions",
                              repeat("longer_seen", longer_repeat), repeat("longer_incremental", longer_repeat),
                              repeat("longer_new", longer_repeat)])
        valid_file = "validation"
        data_dir = os.path.join(base_data_dir, "guidance/")
        task_kwargs = {"batch_size": 64, "k": 3, "max_len": 15, "patience": 5}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = ["nll"]

    elif name == "noisy long lookup":
        task_name = "Noisy Long Lookup Table"
        train_file = "train"
        test_files = flatten(["heldout_inputs", "heldout_compositions", "heldout_tables", "new_compositions",
                              repeat("longer_seen", longer_repeat), repeat("longer_incremental", longer_repeat),
                              repeat("longer_new", longer_repeat)])
        valid_file = "validation"
        data_dir = os.path.join(base_data_dir, "NoisyLongLookupTables/")
        task_kwargs = {"batch_size": 64, "k": 3, "max_len": 30, "patience": 5}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = ["nll"]

    elif name == "symbol rewriting":
        task_name = "Symbol Rewriting"
        train_file = "grammar_std.train.full"
        test_files = ["grammar_long.tst.full", "grammar_repeat.tst.full", "grammar_short.tst.full", "grammar_std.tst.full"]
        valid_file = "grammar.val"
        data_dir = os.path.join(base_data_dir, "SymbolRewriting/")
        task_kwargs = {"batch_size": 64, "k": 3, "max_len": 60, "patience": 15}
        metric_names = ["symbol rewriting accuracy"]
        loss_names = ["nll"]

    elif name == "scan":
        task_name = "SCAN Length"
        train_file = "tasks_train_length"
        test_files = ["tasks_test_length"]
        valid_file = "tasks_validation_length"
        data_dir = os.path.join(base_data_dir, "SCAN/length_split/")
        task_kwargs = {"batch_size": 64, "k": 3, "max_len": 55, "patience": 5}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = ["nll"]

    elif name == "attention lookup":
        task_name = "Attention Lookup"
        train_file = "train"
        test_files = repeat("test_add", 5, start=0)
        valid_file = "validation"
        data_dir = os.path.join(base_data_dir, "AttentionLookup/sample1")
        task_kwargs = {"batch_size": 64, "k": 3, "max_len": 110, "patience": 5}
        metric_names = ["word accuracy", "sequence accuracy", "final target accuracy"]
        loss_names = ["nll"]

    if is_small:
        task_kwargs["k"] = 1

    if is_mini:
        if name == "symbol rewriting":
            train_file = "grammar_std.train.small"
        task_kwargs["k"] = 1
        task_kwargs["batch_size"] = 128
        task_kwargs["patience"] = 2
        task_kwargs["epochs"] = 30

    return Task(task_name, train_file, test_files, valid_file,
                data_dir=data_dir,
                task_kwargs=task_kwargs,
                metric_names=metric_names,
                loss_names=loss_names,)


class Task(object):
    """Helper class containing meta information of datasets.

    Args:
        name (str): name of the dataset.
        train_path (str): path to training data.
        test_paths (list of str): list of paths to all test data.
        valid_path (str, optional): path to validation data.
        extension (str, optional): extension to add to every paths above.
        data_dir (str, optional):  directory to prepend to all path above.
        is_add_to_test (bool, optional): whether to add the train and validation path to the test paths.
        task_kwargs (dictionnaries, optional): list of task specific arguments that update the kwargs for a specific task.
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
                 loss_names=["nll"]):
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

    def _add_presufixes(self, path):
        if path is None:
            return None
        return os.path.join(self.data_dir, path) + self.extension
