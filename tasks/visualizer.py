import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pandas.tools.plotting import table

import torch

from seq2seq.util.helpers import check_import
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.metrics.metrics import get_metrics
from seq2seq.dataset.helpers import get_tabular_data_fields, get_single_data


class AttentionVisualizer(object):
    def __init__(self, task_path,
                 figsize=(13, 7),
                 decimals=3,
                 is_show_positioner=True,
                 is_show_evaluation=True,
                 is_predict_eos=True,
                 output_length_key='length',
                 attention_key="attention_score",
                 position_attn_key='position_attention',
                 content_attn_key='content_attention',
                 positional_table_labels={"mu": "mu", "sigma": "sigma", "p_loc": "position_percentage"}):

        check = Checkpoint.load(task_path)
        self.model = check.model
        self.predictor = Predictor(self.model, check.input_vocab, check.output_vocab)
        self.model_name = task_path.split("/")[-2]
        self.figsize = figsize
        self.decimals = decimals
        self.is_show_positioner = is_show_positioner
        self.is_show_evaluation = is_show_evaluation
        self.is_predict_eos = is_predict_eos
        self.positional_table_labels = positional_table_labels

        self.output_length_key = output_length_key
        self.attention_key = attention_key
        self.position_attn_key = position_attn_key
        self.content_attn_key = content_attn_key

        if self.is_show_evaluation:
            self.tabular_data_fields = get_tabular_data_fields(is_predict_eos=self.is_predict_eos)
            dic_data_fields = dict(self.tabular_data_fields)
            src = dic_data_fields["src"]
            tgt = dic_data_fields["tgt"]

            src.vocab = check.input_vocab
            tgt.vocab = check.output_vocab
            tgt.eos_id = tgt.vocab.stoi[tgt.SYM_EOS]
            tgt.sos_id = tgt.vocab.stoi[tgt.SYM_SOS]

            if "symbol rewriting" in task_path.lower():
                metric_names = ["symbol rewriting accuracy"]
            elif self.is_predict_eos:
                metric_names = ["word accuracy", "sequence accuracy"]
            else:
                metric_names = ["word accuracy", "final target accuracy"]
            self.metrics = get_metrics(metric_names, src, tgt, self.is_predict_eos)
            self.tgt = tgt
            self.src = src

    def __call__(self, input_str, target_str=None):
        output_words, other = self.predictor.predict(input_str.split())
        additional = self._format_additional(other)

        title = self.model_name

        if target_str is not None:
            test = get_single_data([input_str, target_str], self.tabular_data_fields)
            targets = {'decoder_output': torch.LongTensor([self.tgt.vocab.stoi[tok] for tok in test.examples[0].tgt]).view(1, -1),
                       'encoder_input': torch.LongTensor([self.src.vocab.stoi[tok] for tok in test.examples[0].src]).view(1, -1)}
            n_target_words = len(target_str.split())
            if n_target_words > 10:
                target_str = " ".join(target_str.split()[:5] + ["..."] + target_str.split()[-5:])
            title += "\n target_str: {} - ".format(target_str)
            for metric in self.metrics:
                metric.eval_batch(other['sequence'][:min(other["length"][0], n_target_words)], targets)
                title += "{}: {:.2g}  ".format(metric.log_name, metric.get_val())

        if self.attention_key not in additional:
            raise ValueError("`{}` not returned by predictor. Make sure the model uses attention.".format(self.attention_key))

        attention = additional[self.attention_key]
        content_attention = additional.get(self.content_attn_key)
        positional_attention = additional.get(self.position_attn_key)

        if self.is_show_positioner and self.position_attn_key in additional:
            table_values = np.stack([np.around(additional[name], decimals=self.decimals)
                                     for name in self.positional_table_labels.values()]).T

            fig, axs = plt.subplots(2, 2, figsize=self.figsize)
            _plot_attention(input_str, output_words, attention, axs[0, 0], is_colorbar=False, title="Final Attention")
            _plot_table(table_values, list(self.positional_table_labels.keys()), axs[0, 1])
            _plot_attention(input_str, output_words, content_attention, axs[1, 0], title="Content Attention")
            _plot_attention(input_str, output_words, positional_attention, axs[1, 1], title="Positional Attention")

        else:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            _plot_attention(input_str, output_words, attention, ax, title="Final Attention")

        plt.suptitle(title, size=13, weight="bold", y=1.07)

        fig.tight_layout()

        return fig

    def _format_additional(self, additional):
        additional = flatten_dict(additional)

        output = dict()
        output[self.output_length_key] = additional.pop(self.output_length_key)[0]

        for k, v in additional.items():
            output[k] = torch.cat(v).detach().numpy().squeeze()[:output[self.output_length_key]]

        return output


def flatten_dict(d):
    items = []
    for k, v in d.items():
        try:
            items.extend(flatten_dict(v).items())
        except AttributeError:
            items.append((k, v))
    return dict(items)


def _plot_table(values, columns, ax, title=None):
    ax.patch.set_visible(False)
    ax.axis('off')
    table = ax.table(cellText=values, colLabels=columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 1.7)

    if title is not None:
        ax.set_title(title, pad=27)


def subset_attention(attn, input_words, output_words, max_row=16, max_col=16):
    n_row, n_col = attn.shape
    if n_row > max_row:
        subset_rows = max_row // 2
        output_words = output_words[:subset_rows] + output_words[-subset_rows:]
        attn = np.concatenate((attn[:subset_rows, :], attn[-subset_rows:, :]), axis=0)
    if n_col > max_col:
        subset_cols = max_col // 2
        input_words = input_words[:subset_cols] + input_words[-subset_cols:]
        attn = np.concatenate((attn[:, :subset_cols], attn[:, -subset_cols:]), axis=1)
    return attn, input_words, output_words


def _plot_attention(input_sentence, output_words, attention, ax, title=None, is_colorbar=True, **kwargs):
    input_words = input_sentence.split(' ')
    attention, input_words, output_words = subset_attention(attention, input_words, output_words, **kwargs)

    cax = ax.matshow(attention, cmap='bone', vmin=0, vmax=1)
    if is_colorbar:
        plt.colorbar(cax, orientation="horizontal", pad=0.1, ax=ax)

    # Set up axes
    ax.grid(False, which='major')
    ax.set_xticklabels([''] + input_words + ['<EOS>'], rotation=0)
    ax.set_yticklabels([''] + output_words)

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
