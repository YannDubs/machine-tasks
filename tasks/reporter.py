"""
Report generator

TO DO:
- clean
- modularize even more
- be more specific in arguments and returns
- add examples
- use logging instead of printing (not doing that for now, because logging doesn't work well on jupyter notebooks)

Contact: Yann Dubois
"""
import os
import logging
import shutil
import itertools
import ast
import pickle
import json
from functools import reduce
from distutils.dir_util import copy_tree

logging.basicConfig(format='%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import matplotlib
if os.environ.get('DISPLAY', '') == '':
    # HAS TO BE BEFORE IMPORTING PYPLOT
    # solving ssh plotting issue
    logger.warning('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torch

from seq2seq.trainer import SupervisedTrainer
from seq2seq.loss.loss import get_losses
from seq2seq.metrics.metrics import get_metrics
from seq2seq.dataset.helpers import get_tabular_data_fields, get_data
from seq2seq.evaluator import Evaluator, Predictor
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.main import train, get_seq2seq_model, get_is_attn_field

from tasks import get_task
from visualizer import (plot_compare, plot_losses, plot_text, plot_results,
                        AttentionVisualizer, visualize_training, AttentionException)


FILENAMES = dict(results="results.csv",
                 histories="histories.csv",
                 other="other.pkl",
                 parameters='train_arguments.txt',
                 report='report.pdf',
                 locator="locator.pt")

### API ###


def generate_multireport(tasks,
                         output_dir,
                         task_kwargs=None,
                         **kwargs):
    """Make a pdf report experiments either by loading them or recomputing them
    on multiple tasks.

    Args:
        tasks (list of Task or str): list of helper objects containing meta information
            of the task. If list of str, then each element should be the name of
            a task which will be given to `get_task`.
        output_dir (str): directory containing the different models.
        task_kwargs (list of dictionnaries, optional): list of task specific arguments
            that update the kwargs for a specific task.
        kwargs:
            Additional arguments to `generate_report` and `train`.

    Returns:
        models (dictionary): dictionary containing the trained model of the last
            run for each task.
        others (dictionary): dictionary containing additional information for the
            last run of each task.
    """
    models = {}
    others = {}
    pdf = None

    for i, task in enumerate(tasks):
        if isinstance(task, str):
            task = get_task(task)

        task_kwarg = task.task_kwargs
        task_kwarg.update(kwargs)
        if task_kwargs is not None:
            task_kwarg.update(task_kwargs[i])

        logger.info("----- TASK : {} -----".format(task.name))
        logger.info("Task Kwargs: {}.".format(task_kwarg))

        models[task.name], pdf, others[task.name] = generate_report(task,
                                                                    output_dir,
                                                                    _is_multiple_tasks=True,
                                                                    _pdf=pdf,
                                                                    **task_kwarg)
    pdf.close()
    return models, others


def generate_report(task,
                    output_dir,
                    name="model",
                    k=5,
                    is_retrain=True,
                    compare_name=None,
                    n_attn_plots=3,
                    is_plot_train=True,
                    is_rm_FalseNone=False,
                    var_not_show=["max_len", "epochs", "src_vocab", "tgt_vocab",
                                  "batch_size", "eval_batch_size", "save_every",
                                  "print_every", "log_level", "cuda_device",
                                  "checkpoint_path", "name_checkpoint", "patience"],
                    _is_multiple_tasks=False,
                    _pdf=None,
                    **kwargs):
    """Make a pdf report experiments either by loading them or recomputing them.

    Args:
        task (Task): helper object containing meta information of the task.
        output_dir (str): directory containing the different models.
        name (str, optional): base name of the method tested, to which we will
            be appending the value of different parameters.
        k (int, optional): number of times to rerun each task.
        is_retrain (bool, optional): whether to retrain or use the previous saved model.
        compare_name (str, optional): name of the model to which to compare to.
            Have to already be saved.
        n_attn_plots (int, optional): number of example to sample and visualize
            from each test set.
        is_plot_train (bool, optional): whether to plot how the averages of some
            intepretable variables change during training.
        is_rm_FalseNone (bool, optional): if `True` whill not show given arguments
            equal to `False` or `None` in the model name.
        var_not_show (list of str): name of the variables in kwargs that should
            not be shown in the model name. Note that the default values
            won't be shown in the name.
        kwargs:
            Additional arguments to `train`.

    Returns:
        model (seq2seq.models.seq2seq.Seq2seq): trained model in the last run.
        other (dictionary): additional information of the last run.
    """
    parameters_show_name = {k: v for k, v in kwargs.items() if k not in var_not_show}
    name = _namer(name, is_rm_FalseNone=is_rm_FalseNone, **parameters_show_name)

    output_path = os.path.join(output_dir, name)
    task_path = os.path.join(output_path, task.name)
    report_path = os.path.join(output_path, FILENAMES["report"])

    if is_retrain:
        if os.path.exists(task_path) and os.path.isdir(task_path):
            shutil.rmtree(task_path)

        model, other = _train_evaluate(task.name,
                                       task.train_path,
                                       task.test_paths,
                                       task.valid_path,
                                       # oneshot_path=task.oneshot_path,
                                       metric_names=task.metric_names,
                                       loss_names=task.loss_names,
                                       output_dir=output_path,
                                       k=k,
                                       is_viz_train=is_plot_train,
                                       mini_plot_args=[task, name, output_dir, is_plot_train],
                                       **kwargs)
    else:
        model = None
        other = dict()

    # Makes PDF report
    model, other, fig_multiple_tasks = plot_report(task, name, output_dir, is_plot_train,
                                                   n_attn_plots,
                                                   compare_name=compare_name,
                                                   is_multiple_tasks=_is_multiple_tasks)

    other["task_path"] = task_path

    if _is_multiple_tasks:
        pdf = report_path if _pdf is None else _pdf
        pdf = figures_to_pdf(pdf,
                             figures=fig_multiple_tasks,
                             is_close=False)
        return model, pdf, other

    return model, other


def dev_predict(task_path, src_str, is_plot=True, attention_target=None):
    """Helper used to visualize and understand why and what the model predicts.

    Args:
        task_path (str): path to the saved task directory containing, amongst
            other, the model.
        src_str (str): source sentence that will be used to predict.
        is_plot (bool, optional): whether to plots the attention pattern.

    Returns:
        out_words (list): decoder predictions.
        other (dictionary): additional information used for predictions.
        test (dictionary): additional information that is only stored in dev mode.
            These can include temporary variables that do not have to be stored in
            `other` but that can still be interesting to inspect.
    """
    check = Checkpoint.load(task_path)
    check.model.set_dev_mode()

    predictor = Predictor(check.model, check.input_vocab, check.output_vocab)
    if attention_target is not None:
        attention_target = attention_target.split()
    out_words, other = predictor.predict(src_str.split(), attention_target=attention_target)

    test = dict()

    for k, v in other["test"].items():
        tensor = v if isinstance(v, torch.Tensor) else torch.cat(v)
        test[k] = tensor.detach().cpu().numpy().squeeze()[:other["length"][0]]
        # except: # for using "step"
        # test[k] = v

    if is_plot:
        visualizer = AttentionVisualizer(task_path)
        if attention_target is not None:
            attention_target = " ".join(attention_target)  # needs to be str
        visualizer(src_str, attention_target=attention_target)

    return out_words, other, test

### Plotting ###


def figures_to_pdf(file, figures, is_close=True):
    """Plots every given figure as a new page of the pdf.

    Args:
        file (str or PdfPages): file path where to save the PDF.
        figures (list): list of matplotlib figures, each of them will be saved as
            a new PDF page.
        is_close (bool, optional): whether to close the PDF file or to return it.
            The Latter is useful if you want to save something in the PDF later.
    """
    if isinstance(file, str):
        if os.path.isfile(file):
            try:
                os.rename(file, " ".join(file.split(".")[:-1]) + "_old.pdf")
            except OSError as e:
                print(e)
                pass
        pdf = PdfPages(file)
    else:
        pdf = file

    for fig in figures:
        pdf.savefig(fig)
        plt.close()

    if is_close:
        pdf.close()
    else:
        return pdf


def _generate_attn_figs(files, task_path, n_sample_plots=3, **kwargs):
    """Generates `n_sample_plots` attention plots by sampling examples in `files`
        and predicting using the model in `task_path`.
    """
    attn_visualizer = AttentionVisualizer(task_path, is_show_name=False, **kwargs)

    def generator():
        """Uses a sub generator in order to correctly catch errors of the
        `AttentionVisualizer` constructor.
        """
        for file in files:
            try:
                samples = pd.read_csv(file,
                                      sep="\t",
                                      header=None,
                                      usecols=[0, 1, 2]).sample(n_sample_plots)
            except ValueError:
                samples = pd.read_csv(file,
                                      sep="\t",
                                      header=None,
                                      usecols=[0, 1]).sample(n_sample_plots)

            yield plot_text(file.split("/")[-1])

            for _, sample in samples.iterrows():
                attention_target = sample[2] if len(sample) > 2 else None
                attn_fig = attn_visualizer(sample[0],
                                           tgt_str=sample[1],
                                           attention_target=attention_target)
                yield attn_fig

    return generator()


def plot_report(task, name, output_dir, is_plot_train, n_attn_plots,
                compare_name=None,
                sub_run=None,
                is_multiple_tasks=False):

    output_path = os.path.join(output_dir, name)
    task_name = task.name if sub_run is None else "{}_{}".format(task.name, sub_run)
    task_path = os.path.join(output_path, task_name)
    report_task_path = os.path.join(task_path, FILENAMES["report"])

    (model, results, histories,
     other, parameters) = _load_output_training(task_path)

    k = histories["k"].max() + 1

    # TITLE #
    to_format = "{} \n Task: {} \n # parameters : {} \n # runs : {}"
    text_title = to_format.format(task.name,
                                  name.capitalize(),
                                  parameters["n_parameters"],
                                  k)
    fig_title = plot_text(text_title, size=10)

    # MODEL #
    fig_model = plot_text(str(model), size=6, ha="left", x=0.2)

    # LOSSES #
    fig_losses = plot_losses(histories,
                             title="{} - training and validation losses".format(task.name))

    # METRICS #
    title_results = '{} - average metrics. Bootstrap 95 % CI.'.format(task.name)

    if compare_name is not None:
        compare_to = reduce(os.path.join,
                            [output_dir, compare_name, task.name, FILENAMES["results"]])
        compare_to = pd.read_csv(compare_to)
        grid = plot_compare(results, compare_to, name, compare_name,
                            is_plot_mean=True,
                            title=title_results)
    else:
        grid = plot_results(results, is_plot_mean=True, title=title_results)
    fig_results = grid.fig

    figs_generator = [fig_title, fig_model, fig_losses, fig_results]

    # VIZ TRAIN #
    if is_plot_train:
        figs_generator += visualize_training(other["visualize"], model)

    # ATTENTION #
    if n_attn_plots != 0:
        try:
            attn_figs_generator = _generate_attn_figs(task.test_paths, task_path,
                                                      n_sample_plots=n_attn_plots,
                                                      is_predict_eos=parameters["is_predict_eos"])
        except AttentionException:
            logger.warning("Skipping the attention plotter because the model is not using attention.")
            attn_figs_generator = []

        # generator because could be very memory heavy
        figs_generator = itertools.chain(figs_generator, attn_figs_generator)

    # RETURN #

    figures_to_pdf(report_task_path, figures=figs_generator)

    if is_multiple_tasks:
        return model, other, [fig_title, fig_model, fig_losses, fig_results]
    else:
        return model, other


### Helpers ###


def _namer(name, is_rm_FalseNone=False, **kwargs):
    """Append variable name and value in alphabetical order to `name`."""
    def _key_value_to_name(k, v):
        if isinstance(v, list):
            return "_".join(sorted(v))
        if isinstance(v, bool):
            if not v:
                return k.replace("is_", "no_")
            else:
                return k.split("is_")[-1]
        elif isinstance(v, str):
            return v
        else:
            return k.split("_size")[0] + str(v)

    if is_rm_FalseNone:
        kwargs = {k: v for k, v in kwargs.items() if v != False and v is not None}
    kwargs = sorted([_key_value_to_name(k, v) for k, v in kwargs.items()])
    if kwargs != {}:
        name += "_" + "_".join(kwargs)
    return name


def _format_losses_history(histories):
    """Formats a pd.DataFrame where the `k` rows correspond to different runs,
    the columns are the different datas and the each cell contains a list of `n_epoch`
    values corresponding to the loss at each epoch. Return a melted pd.dataframe
    with `n_epoch` rows and `columns=["epoch","k","loss","data"]`.
    """
    histories = {name: pd.DataFrame(dict(zip(col.index, col.values)))
                 for name, col in histories.iteritems()}
    losses = []
    for name, df in histories.items():
        value_col = df.columns
        df['epoch'] = df.index
        df = pd.melt(df, value_vars=value_col, var_name="k", value_name="loss", id_vars="epoch")
        df["data"] = name.split('_')[-1]
        losses.append(df)
    losses = pd.concat(losses, axis=0)
    return losses


def _tensors_to_np_array(list_tensors):
    """COnversts a list of tensors to a numpy array."""
    try:
        arr = torch.stack(list_tensors).detach().squeeze().cpu().numpy()
    except TypeError:
        arr = np.array(list_tensors)
    arr = np.atleast_1d(arr)

    return arr


def _format_other(other):
    """Format the additional outputs of the model predictions."""
    return {k_ext: {k_int: _tensors_to_np_array(v_int) for k_int, v_int in v_ext.items()}
            for k_ext, v_ext in other.items()}


def _load_output_training(task_path):
    """Loads all the components that were saved during and at the end of training."""
    checkpoint = Checkpoint.load(task_path)
    model = checkpoint.model
    results = pd.read_csv(os.path.join(task_path, FILENAMES["results"]))
    histories = pd.read_csv(os.path.join(task_path, FILENAMES["histories"]),
                            converters={0: ast.literal_eval, 1: ast.literal_eval})
    with open(os.path.join(task_path, FILENAMES["other"]), 'rb') as f:
        other = pickle.load(f)
    with open(os.path.join(task_path, FILENAMES["parameters"]), 'r') as f:
        parameters = json.load(f)

    histories = _format_losses_history(histories)

    return model, results, histories, other, parameters

### Evalation ###


def _evaluate(checkpoint_path, test_paths,
              metric_names=["word accuracy", "sequence accuracy", "final target accuracy"],
              loss_names=["nll"],
              max_len=50,
              batch_size=32,
              is_predict_eos=True,
              attender=None,
              force_mu=None):
    """Evaluates the models saved in a checkpoint."""
    results = []

    print("loading checkpoint from {}".format(os.path.join(checkpoint_path)))
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model

    tabular_data_fields = get_tabular_data_fields(is_predict_eos=is_predict_eos,
                                                  is_add_attn=get_is_attn_field(attender, loss_names, metric_names, force_mu))

    dic_data_fields = dict(tabular_data_fields)
    src = dic_data_fields["src"]
    tgt = dic_data_fields["tgt"]

    src.vocab = checkpoint.input_vocab
    tgt.vocab = checkpoint.output_vocab
    tgt.eos_id = tgt.vocab.stoi[tgt.SYM_EOS]
    tgt.sos_id = tgt.vocab.stoi[tgt.SYM_SOS]

    for test_path in test_paths:
        test = get_data(test_path, max_len, tabular_data_fields)

        metrics = get_metrics(metric_names, src, tgt, is_predict_eos)
        losses, loss_weights = get_losses(loss_names, tgt, is_predict_eos)

        evaluator = Evaluator(loss=losses, batch_size=batch_size, metrics=metrics)
        data_func = SupervisedTrainer.get_batch_data
        losses, metrics = evaluator.evaluate(model=seq2seq, data=test, get_batch_data=data_func)

        total_loss, log_msg, _ = SupervisedTrainer.get_losses(losses, metrics, 0)

        dataset = test_path.split('/')[-1].split('.')[0]
        results.append([dataset, total_loss] + [metric.get_val() for metric in metrics])

    results_df = pd.DataFrame(results,
                              columns=["Dataset", "Loss"] + [metric.name for metric in metrics])

    results_df = results_df.melt(id_vars=['Dataset'], var_name="Metric", value_name='Value')

    return results_df


def _save_output_training(output_path, results, other, histories, model):
    results.to_csv(os.path.join(output_path, FILENAMES["results"]),
                   index=False)
    histories.to_csv(os.path.join(output_path, FILENAMES["histories"]),
                     index=False)
    with open(os.path.join(output_path, FILENAMES["other"]), 'wb') as f:
        pickle.dump(other, f)

    try:
        file = os.path.join(output_path, FILENAMES["locator"])
        model.decoder.attender.save_locator(file)
    except AttributeError as e:
        print(e)
        pass


def _train_evaluate(name,
                    train_path,
                    test_paths,
                    valid_path,
                    # oneshot_path=None,
                    metric_names=["word accuracy", "sequence accuracy", "final target accuracy"],
                    loss_names=["nll"],
                    k=5,
                    output_dir="models/",
                    max_len=50,
                    is_save=True,
                    is_predict_eos=True,
                    batch_size=32,
                    mini_plot_args=[],
                    attender=None,
                    force_mu=None,
                    **kwargs):
    """Train a model and evaluate it."""
    output_path = os.path.join(output_dir, name)

    if is_predict_eos and (force_mu is not None or attender == "hard"):
        is_predict_eos = False
        logging.warning("Setting `is_predict_eos=False` because `is_attn_field`.")

    if not is_predict_eos:
        if "final target accuracy" not in metric_names:
            # final target accuracy is like sequence accuracy but is correct if
            # outputted too many token (but not if not enough!)
            metric_names.append("final target accuracy")

    results_dfs = [None] * k
    histories = [None] * k
    is_last_run = False
    for i in range(k):
        if i % 5 == 0:
            logging.info("run: {}".format(i))

        if i == k - 1:
            is_last_run = True

        model, history, other = train(train_path, valid_path,
                                      # oneshot_path=oneshot_path,
                                      output_dir=output_dir,
                                      max_len=max_len,
                                      name_checkpoint=name,
                                      is_predict_eos=is_predict_eos,
                                      batch_size=batch_size,
                                      attender=attender,
                                      metric_names=metric_names,
                                      loss_names=loss_names,
                                      force_mu=force_mu,
                                      **kwargs)

        other = _format_other(other)
        histories[i] = list(history)
        results_dfs[i] = _evaluate(output_path, test_paths,
                                   is_predict_eos=is_predict_eos,
                                   max_len=max_len,
                                   batch_size=batch_size,
                                   attender=attender,
                                   metric_names=metric_names,
                                   loss_names=loss_names,
                                   force_mu=force_mu)

        if k > 1:
            i_output_path = output_path + "_{}".format(i)
            copy_tree(output_path, i_output_path)

            if is_save:
                histories_i = pd.DataFrame([histories[i]], columns=history.names)
                _save_output_training(i_output_path, results_dfs[i], other, histories_i,
                                      model)

            # making smaller report for each run
            plot_report(*mini_plot_args,
                        n_attn_plots=1,
                        sub_run=i)

        if not is_last_run:
            shutil.rmtree(output_path)

    histories = pd.DataFrame(histories, columns=history.names)
    results = pd.concat(results_dfs).reset_index(drop=True)

    if is_save:
        _save_output_training(output_path, results, other, histories,
                              model)

    return model, other
