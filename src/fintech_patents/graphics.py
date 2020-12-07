# coding=utf-8
# Copyright 2020 George Mihaila.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Deal with webapp graphics components."""

import html
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
# Added fix from https://docs.streamlit.io/en/stable/deploy_streamlit_app.html
_lock = RendererAgg.lock


def html_highlight_text(weights, tokens, color=(135, 206, 250), intensity=1):
    """
    Return HTML code of highlighted tokens with different intensity color.

    Arguments:
        weights: (`obj`:list):
            List of attentions scores for each token.

        tokens: (`obj`:list):
            List of tokens used in input. Need to match in length with attentions.

        color: (`obj`: str):
            String of RGB values separated by comma.

        intensity: (`obj`: int):
            Multiply each weight by this value to show up color.

    Returns:
        (`obj`: str):
            String containing html code for color highlight.
    """

    # Store all text with html code for coloring.
    highlighted_text = []
    # Larges weight value used to normalize.
    max_weight_value = 1 if None in weights else max(weights)
    # Loop through each weight and token to highlight with color.
    for weight, word in zip(weights, tokens):
        # Append html code text if we have weight.
        if weight is not None:
            highlighted_text.append(
                f'<span style="background-color:rgba({str(color)[1:-1]},\
                {str((intensity * weight) / max_weight_value)});">{html.escape(word)}</span>')
        else:
            # Append plain text if no weight.
            highlighted_text.append(word)

    # Return single string with html code.
    return ' '.join(highlighted_text)


def plot_labels_confidence(labels_percentages, labels_coloring):
    # Figure Size
    with _lock:
        plt.rcParams.update({'font.size': 22})
        fig, ax = plt.subplots(figsize=(16, 9))

        # Make the plot
        for index, (label, percent) in enumerate(labels_percentages.items()):
            plt.barh(index, percent, color=np.array(labels_coloring[label]) / 255,
                     edgecolor='grey', label=label)

        plt.yticks(list(range(len(labels_percentages) + 1)), labels_percentages.keys())

        # Add padding between axes and labels
        ax.xaxis.set_tick_params(pad=5)
        ax.yaxis.set_tick_params(pad=10)

        # Add x, y gridlines
        ax.grid(b=True, color='grey',
                linestyle='-.', linewidth=0.5,
                alpha=0.2)

        # Show top values
        ax.invert_yaxis()

        # Add annotation to bars
        for i in ax.patches:
            plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
                     f'{round((i.get_width()), 2)}%',
                     fontsize=22, fontweight='bold',
                     color='grey')

        plt.xticks(list(range(0, 110, 10)))
        ax.get_xaxis().set_visible(False)
        plt.box(False)

        return fig
