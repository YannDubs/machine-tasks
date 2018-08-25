"""Module containing all the callables that return figures."""

import warnings

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pandas.tools.plotting import table

import torch

from seq2seq.util.helpers import check_import
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.metrics.metrics import get_metrics
from seq2seq.dataset.helpers import get_tabular_data_fields, get_single_data


def _plot_mean(data, **kwargs):
    """Plot a horizontal line for the mean"""
    m = data.mean()
    plt.axhline(m, **kwargs)


def plot_compare(df1, df2, model1, model2, **kwargs):
    """Plot and compares evaluation results of 2 models."""
    df1['Model'] = model1
    df2['Model'] = model2
    df = pd.concat([df1, df2])
    return plot_results(df, **kwargs)


def plot_losses(losses, title="Training and validation losses"):
    """Plots the losses given a pd.dataframe with `n_epoch` rows and `columns=["epoch","k","loss","data"]`."""
    sns.set(font_scale=1.5, style="white")
    f, ax = plt.subplots(figsize=(11, 7))
    grid = sns.tsplot(time="epoch",
                      value="loss",
                      unit="k",
                      err_style="unit_traces",
                      condition="data",
                      data=losses,
                      ax=ax)
    sns.despine()
    if title is not None:
        grid.set_title(title)
    return f


def plot_text(txt, size=12, ha='center', **kwargs):
    """Plots text as an image and returns the matplotlib figure."""
    fig = plt.figure(figsize=(11, 7))
    text = fig.text(0.5, 0.5, txt, ha=ha, va='center', size=size, **kwargs)
    return fig


def plot_results(data, is_plot_mean=False, title=None, **kwargs):
    """Plot evaluation results.

    Args:
        data (pd.DataFrame): dataframe containing the losses and metric results
            in a "Dataset", a "Value", a "Metric", and an optional "Model" column.
        is_plot_mean (bool, optional): whether to plot the mean value with a horizontal bar.
        title (str, optional): title to add.
        kwargs:
            Additional arguments to `sns.factorplot`.
    """
    sns.set(font_scale=1.5)
    hue = "Model" if "Model" in data.columns and data['Model'].nunique() > 1 else None
    grid = sns.factorplot(x="Dataset",
                          y="Value",
                          col="Metric",
                          kind="bar",
                          size=9,
                          sharey=False,
                          ci=95,
                          hue=hue,
                          data=data,
                          **kwargs)
    grid.set_xticklabels(rotation=90)
    for ax in grid.axes[0, 1:]:
        ax.set_ylim(0, 1)
    if is_plot_mean:
        grid.map(_plot_mean, 'Value', ls="--", c=".5", linewidth=1.3)
    if title is not None:
        plt.subplots_adjust(top=0.9)
        grid.fig.suptitle(title)
    return grid

### ATTENTION VISUALISATION ###


class AttentionException(Exception):
    """Custom exception indicating an issue with the attention pattern."""


class MetricComputer(object):
    """Object for computing metrics for a given task.

    Args:
        checkpoint (str or seq2seq.utils.Checkpoint): checkpoint object or the
            name of file that can be loasded as one, containing the model that
            should be used to get the source and target vocabulary.
        is_predict_eos (bool, optional): whether to predict the <eos> token.
        is_symbol_rewriting (bool, optional): whether the task is symbol rewriting.
    """

    def __init__(self, checkpoint, is_predict_eos=True, is_symbol_rewriting=False):
        if isinstance(checkpoint, str):
            checkpoint = Checkpoint.load(checkpoint)

        self.is_predict_eos = is_predict_eos
        self.tabular_data_fields = get_tabular_data_fields(is_predict_eos=self.is_predict_eos)
        dic_data_fields = dict(self.tabular_data_fields)
        src = dic_data_fields["src"]
        tgt = dic_data_fields["tgt"]

        src.vocab = checkpoint.input_vocab
        tgt.vocab = checkpoint.output_vocab
        tgt.eos_id = tgt.vocab.stoi[tgt.SYM_EOS]
        tgt.sos_id = tgt.vocab.stoi[tgt.SYM_SOS]

        if is_symbol_rewriting:
            metric_names = ["symbol rewriting accuracy"]
        elif self.is_predict_eos:
            metric_names = ["word accuracy", "sequence accuracy"]
        else:
            metric_names = ["word accuracy", "final target accuracy"]
        self.metrics = get_metrics(metric_names, src, tgt, self.is_predict_eos)
        self.tgt = tgt
        self.src = src

    def __call__(self, src_str, out_str, tgt_str):
        """Compute and return a dictionary of computed metrics.

        Args:
            src_str (str): source sentence of the example.
            out_str (str): prediction of the example.
            tgt_str (str): target of the example.
        """
        for metric in self.metrics:
            metric.reset()

        n_target_words = len(tgt_str.split())
        test = get_single_data([src_str, tgt_str], self.tabular_data_fields)
        decoder_outputs = [self.tgt.vocab.stoi[tok] for tok in test.examples[0].tgt]
        encoder_input = [self.src.vocab.stoi[tok] for tok in test.examples[0].src]
        targets = {'decoder_output': torch.LongTensor(decoder_outputs).view(1, -1),
                   'encoder_input': torch.LongTensor(encoder_input).view(1, -1)}

        outputs = [torch.tensor(self.tgt.vocab.stoi[tok]).view(1, -1)
                   for tok in out_str.split()]
        outputs = outputs[:min(len(out_str.split()), n_target_words + 1)]
        for metric in self.metrics:
            metric.eval_batch(outputs, targets)

        metric_dict = {metric.log_name: metric.get_val()
                       for metric in self.metrics}

        return metric_dict


class AttentionVisualizer(object):
    """Object for visualizing the attention pattern of a given prediction.

    Args:
        task_path (str): name of the  checkpoint file.
        figsizeh (tuple, optional): (width, height) of the final matplotlib figure.
        decimals (int, optional): number of decimals to whoe when pritning any number.
        is_show_attn_split (bool, optional): whether to show the the content and
            positional attention if there is one in addition to the full attention.
        is_show_evaluation (bool, optional): whether to show the evaluation metric
            if the target is given.
        output_length_key, attention_key, content_attn_key (str, optional): keys of
            the respective values in the the dictionary returned by the prediction.
        positional_table_labels (dictionary, optional): mapping from the keys
            in the return dictionary (the values) to the name the name of it should
            be shown as in the figure (the keys).
        is_show_name (bool, optional): whether to show the name of the mdoel as
            the title of the figure.
        max_src, max_out, max_tgt (int, optional): maximum number of token to show
            for the source, the output and the target. Used in order no to clotter
            too much the plots.
        kwargs:
            Additional arguments to `MetricComputer`.
    """

    def __init__(self, task_path,
                 figsize=(13, 13),
                 decimals=2,
                 is_show_attn_split=True,
                 is_show_evaluation=True,
                 output_length_key='length',
                 attention_key="attention_score",
                 position_attn_key='position_attention',
                 content_attn_key='content_attention',
                 positional_table_labels={"μ": "mu",
                                          "σ": "sigma",
                                          "λ%": "position_percentage",
                                          "C.γ": "content_confidence",
                                          "C.λ": "pos_confidence",
                                          "w_μ": "mu_old_weight",
                                          "w_α": "mean_attn_old_weight",
                                          "w_j": "rel_counter_decoder_weight",
                                          "w_1/n": "single_step_weight",
                                          "w_γ": "mean_content_old_weight",
                                          "w_1": "bias_weight"},
                 # "% carry": "carry_rates",
                 is_show_name=True,
                 max_src=17,
                 max_out=13,
                 max_tgt=13,
                 **kwargs):

        check = Checkpoint.load(task_path)
        self.model = check.model
        # store some interesting variables
        self.model.set_dev_mode()

        self.predictor = Predictor(self.model, check.input_vocab, check.output_vocab)
        self.model_name = task_path.split("/")[-2]
        self.figsize = figsize
        self.decimals = decimals
        self.is_show_attn_split = is_show_attn_split
        self.is_show_evaluation = is_show_evaluation
        self.positional_table_labels = positional_table_labels
        self.is_show_name = is_show_name

        self.max_src = max_src
        self.max_out = max_out
        self.max_tgt = max_tgt

        self.output_length_key = output_length_key
        self.attention_key = attention_key
        self.position_attn_key = position_attn_key
        self.content_attn_key = content_attn_key

        if self.is_show_evaluation:
            self.is_symbol_rewriting = "symbol rewriting" in task_path.lower()
            self.metric_computer = MetricComputer(check,
                                                  is_symbol_rewriting=self.is_symbol_rewriting,
                                                  **kwargs)

        if self.model.decoder.use_attention is None:
            raise AttentionException("Model is not using attention.")

    def __call__(self, src_str, tgt_str=None):
        """Plots the attention for the current example.

        Args:
            src_str (str): source of the example.
            tgt_str (str, optional): (width, height) target of the example,
                must be given in order to show the final metric.
        """
        out_words, other = self.predictor.predict(src_str.split())

        full_src_str = src_str
        full_out_str = " ".join(out_words)
        full_tgt_str = tgt_str

        additional, additional_text = self._format_additional(other)
        additional, src_words, out_words, tgt_str = self._subset(additional,
                                                                 src_str.split(),
                                                                 out_words,
                                                                 tgt_str)

        if self.is_show_name:
            title = ""
        else:
            title = None

        if tgt_str is not None:
            if self.is_show_name:
                title += "\n tgt_str: {} - ".format(tgt_str)
            else:
                title = "tgt_str: {} - ".format(tgt_str)

            if self.metric_computer.is_predict_eos:
                is_output_good_length = (len(full_out_str.split()) != len(full_tgt_str.split()))
                if self.is_symbol_rewriting and is_output_good_length:
                    warnings.warn("Cannot currently show the metric for symbol rewriting if output is not the right length.")

                else:
                    metrics = self.metric_computer(full_src_str, full_out_str, full_tgt_str)

                    for name, val in metrics.items():
                        title += "{}: {:.2g}  ".format(name, val)
            else:
                warnings.warn("Cannot currently show the metric in the attention plots when `is_predict_eos=False`")

        if self.attention_key not in additional:
            raise ValueError("`{}` not returned by predictor. Make sure the model uses attention.".format(self.attention_key))

        attention = additional[self.attention_key]

        filtered_pos_table_labels = {k: v for k, v in self.positional_table_labels.items()
                                     if v in additional}
        table_values = np.stack([np.around(additional[name], decimals=self.decimals)
                                 for name in filtered_pos_table_labels.values()]).T

        if self.is_show_attn_split and (self.position_attn_key in additional
                                        and self.content_attn_key in additional):
            content_attention = additional.get(self.content_attn_key)
            positional_attention = additional.get(self.position_attn_key)

            fig, axs = plt.subplots(2, 2, figsize=self.figsize)
            _plot_attention(src_words, out_words, attention, axs[0, 0],
                            is_colorbar=False,
                            title="Final Attention")
            _plot_table(table_values, list(filtered_pos_table_labels.keys()), axs[0, 1])
            _plot_attention(src_words, out_words, content_attention, axs[1, 0],
                            title="Content Attention")
            _plot_attention(src_words, out_words, positional_attention, axs[1, 1],
                            title="Positional Attention")

        elif self.position_attn_key in additional:
            fig, axs = plt.subplots(1, 2, figsize=self.figsize)
            _plot_attention(src_words, out_words, attention, axs[0], title="Final Attention")
            _plot_table(table_values, list(filtered_pos_table_labels.keys()), axs[1])
        else:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            _plot_attention(src_words, out_words, attention, ax, title="Final Attention")

        fig.text(0.5, 0.02, ' | '.join(additional_text), ha='center', va='center', size=13)

        if title is not None:
            plt.suptitle(title, size=13, weight="bold")
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.07, top=0.83)

        return fig

    def _format_additional(self, additional):
        """Format the additinal dictionary returned by the predictor."""
        def _format_carry_rates(carry_rates):
            if carry_rates is None:
                return "Carry % : None"
            mean_carry_rates = np.around(carry_rates.mean().item(), decimals=self.decimals)
            median_carry_rates = np.around(carry_rates.median().item(), decimals=self.decimals)
            return "Carry % : mean: {}; median: {}".format(mean_carry_rates, median_carry_rates)

        def _format_mu_weights(mu_weights):
            if mu_weights is not None:
                building_blocks_labels = self.model.decoder.position_attention.building_blocks_labels
                for i, label in enumerate(building_blocks_labels):
                    output[label + "_weight"] = mu_weights[:, i]

        output = dict()

        additional.pop("visualize", None)  # this is only for training visualization not predict
        additional.pop("losses", None)

        additional_text = []
        additional = flatten_dict(additional)

        output = dict()
        output[self.output_length_key] = additional.pop(self.output_length_key)[0]

        for k, v in additional.items():
            tensor = v if isinstance(v, torch.Tensor) else torch.cat(v)
            output[k] = tensor.detach().cpu().numpy().squeeze()[:output[self.output_length_key]]

        carry_txt = _format_carry_rates(additional.pop("carry_rates", None))
        additional_text.append(carry_txt)

        _format_mu_weights(output.pop("mu_weights", None))

        return output, additional_text

    def _subset(self, additional, src_words, out_words, tgt_str=None):
        """Subsets the objects in the additional dictionary in order not to
        clotter the visualization.
        """
        n_src = len(src_words)
        n_out = len(out_words)

        if n_out > self.max_out:
            subset_out = self.max_out // 2
            out_words = out_words[:subset_out] + out_words[-subset_out:]
            for k, v in additional.items():
                if isinstance(v, np.ndarray):
                    additional[k] = np.concatenate((v[:subset_out], v[-subset_out:]),
                                                   axis=0)

        if n_src > self.max_src:
            subset_src = self.max_src // 2
            src_words = src_words[:subset_src] + src_words[-subset_src:]
            for k, v in additional.items():
                if isinstance(v, np.ndarray) and v.ndim == 2:
                    additional[k] = np.concatenate((v[:, :subset_src], v[:, -subset_src:]),
                                                   axis=1)

        if tgt_str is not None:
            tgt_words = tgt_str.split()
            n_tgt = len(tgt_words)
            if n_tgt > self.max_tgt:
                subset_target = self.max_tgt // 2
                tgt_str = " ".join(tgt_words[:subset_target] +
                                   ["..."] +
                                   tgt_words[-subset_target:])

        return additional, src_words, out_words, tgt_str


def _plot_table(values, columns, ax, title=None):
    """Plots a table as a figure."""
    ax.patch.set_visible(False)
    ax.axis('off')
    table = ax.table(cellText=values, colLabels=columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.17, 1.7)

    if title is not None:
        ax.set_title(title, pad=27)


def _plot_attention(src_words, out_words, attention, ax, title=None, is_colorbar=True):
    """Plots a matrix containing the attention pattern."""
    cax = ax.matshow(attention, cmap='bone', vmin=0, vmax=1)
    if is_colorbar:
        plt.colorbar(cax, orientation="horizontal", pad=0.1, ax=ax)

    # Set up axes
    ax.grid(False, which='major')
    ax.set_xticklabels([''] + src_words + ['<EOS>'], rotation=0)
    ax.set_yticklabels([''] + out_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.set_xticks([x - 0.5 for x in ax.get_xticks()][1:], minor='true')
    ax.set_yticks([y - 0.5 for y in ax.get_yticks()][1:], minor='true')
    ax.grid(which='minor', linestyle='dotted')

    # ax.set_xlabel("INPUT")
    ax.set_ylabel("OUTPUT")
    ax.yaxis.set_label_position('left')
    # ax.xaxis.set_label_position('top')
    #ax.xaxis.labelpad = 9

    if title is not None:
        ax.set_title(title, pad=27)

### TRAINING VISUALISATION ###


def _plot_training_phase(df, title=None, **kwargs):
    """Plots a melted dataframe containing the training history of interpretable
     variables."""
    sns.set(font_scale=1.5)
    grid = sns.factorplot(data=df,
                          x="epochs",
                          y="value",
                          margin_titles=True,
                          sharey=True,
                          sharex=True,
                          scale=0.5,
                          **kwargs)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10, integer=True))

    if title is not None:
        plt.subplots_adjust(top=0.9)
        grid.fig.suptitle(title)

    return grid


def _plot_building_blocks(to_visualize, labels,
                          title="Training Process - Positioning Attention Building Blocks"):
    """Plots the positioning building blocaks and their weights.

    Args:
        to_visualize (dictionary): dictionary containing all the variables to visualize.
            These variables are averaged over decoding step, over batch_size, and
            over all batches. But they are not averaged over epochs.building_blocks.
        labels (list): building_blocks labels in order.
        title (str, optional): title to add.
    """
    building_blocks = pd.DataFrame(to_visualize["building_blocks"],
                                   columns=labels
                                   ).reset_index().melt(id_vars="index")

    mu_weights = pd.DataFrame(to_visualize["mu_weights"],
                              columns=labels
                              ).reset_index().melt(id_vars="index")
    building_blocks["mu_weights"] = mu_weights["value"]

    value_vars = ["mu_weights", "building_blocks"]

    building_blocks = building_blocks.rename(columns={"value": "building_blocks",
                                                      "index": "epochs",
                                                      "variable": "block"})
    building_blocks = building_blocks.melt(id_vars=["epochs", "block"],
                                           value_vars=value_vars)

    grid = _plot_training_phase(building_blocks,
                                title=title,
                                col="block",
                                row="variable")

    return grid


def _plot_vis_train_no_bb(to_visualize,
                          title="Training Process - Averages of Interprtable Variables."):
    """Plots all interpretable variables change during training besides the
    building blocks related ones.

    Args:
        to_visualize (dictionary): dictionary containing all the variables to visualize.
            These variables are averaged over decoding step, over batch_size, and
            over all batches. But they are not averaged over epochs.
        title (str, optional): title to add.
    """
    keys_to_rm = ["mu_weights", 'sigma_weights', "building_blocks"]
    to_visualize_no_bb = rm_dict_keys(to_visualize, keys_to_rm)
    to_visualize_no_bb = pd.DataFrame(to_visualize_no_bb)
    to_visualize_no_bb = to_visualize_no_bb.reset_index()
    to_visualize_no_bb = to_visualize_no_bb.melt(id_vars="index")
    to_visualize_no_bb = to_visualize_no_bb.rename(columns={"index": "epochs"})

    grid = _plot_training_phase(to_visualize_no_bb,
                                title=title,
                                col_wrap=4,
                                col="variable")

    return grid


def visualize_training(to_visualize, model):
    """Return list of plots of interpretable variables change during training.

    Args:
        to_visualize (dictionary): dictionary containing all the variables to visualize.
            These variables are averaged over decoding step, over batch_size, and
            over all batches. But they are not averaged over epochs.
        model (seq2seq.Seq2seq): trained model.
        title (str, optional): title to add.
    """
    to_return = []

    if len(to_visualize) == 0:
        return to_return

    grid_no_building_blocks = _plot_vis_train_no_bb(to_visualize)
    to_return.append(grid_no_building_blocks.fig)

    if "building_blocks" in to_visualize:
        building_blocks_labels = model.decoder.position_attention.building_blocks_labels
        grid_building_blocks = _plot_building_blocks(to_visualize, building_blocks_labels)
        to_return.append(grid_building_blocks.fig)

    return to_return

# Helpers


def rm_dict_keys(dic, keys_to_rm):
    """remove a set of keys from a dictionary not in place."""
    return {k: v for k, v in dic.items() if k not in keys_to_rm}


def flatten_dict(d):
    """Flattens a dictionary."""
    items = []
    for k, v in d.items():
        try:
            items.extend(flatten_dict(v).items())
        except AttributeError:
            items.append((k, v))
    return dict(items)
