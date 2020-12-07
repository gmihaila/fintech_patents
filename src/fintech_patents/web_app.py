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
"""Web app implementation using Streamlit."""

import os
import io
import argparse
import configparser
import streamlit as st
from pickle_models import pickle_pytorch_models
from downloads_models import download_from_config
from inference_modeling import (inference_transformer,
                                )
from graphics import (html_highlight_text,
                      plot_labels_confidence,
                      )
from settings import (CONFIG_FILE, IDS_LABELS, LABELS_COLORS,
                      SAMPLE_ABSTRACT,
                      )
import psutil
import sys
import gc


def app_header():
    r"""
    Here is where title, subtitle and app description will go
    """

    # Title
    st.title('FinTech Patent Classification')
    # Subtitle
    # st.subheader('Make predictions')
    # Description
    # st.write('More details go here...')
    
#     st.write('/ncheck resouces')
#     st.write('python verison')
#     st.write(sys.version)
    
#     st.write('virtual_memory')
#     st.write(psutil.virtual_memory())
    
#     st.write('disk_partitions')
#     st.write(psutil.disk_partitions())
    
#     st.write('disk_usage')
#     st.write(psutil.disk_usage('/'))
    
#     st.write('end resouces/n')

    return


def app_modeling(config_file):
    r"""
    This is where the modeling of the app happens.

    """

    models_display_names = [config[sec]['display_name'] for sec in config_file.sections()]
    model_selected = st.selectbox('Choose Model', list(models_display_names))

    model_description = [config_file[sec]['description'] for sec in config_file.sections() if
                         config_file[sec]['display_name'] == model_selected][0]

    model_tokenizer_pickle_path = [config_file[sec]['model_tokenizer_pickle_path'] for sec in config_file.sections() if
                                   config_file[sec]['display_name'] == model_selected][0]

    st.markdown(f'Current Model Selected: **{model_selected}**')

    st.markdown(f'Model description: *{model_description}*')

    if os.path.isfile(SAMPLE_ABSTRACT):
        default_patent = io.open(SAMPLE_ABSTRACT, mode='r', encoding='utf-8').read()
    else:
        default_patent = ''

    st.markdown('### Patent Text:')
    user_input = st.text_area(label="Copy&Paste here:", value=default_patent, height=400)

    st.markdown('### Attention intensity')
    intensity = st.slider(label='Intensity of text color highlight in predictions:',
                          min_value=1, max_value=100, value=1)

    if st.button('Get Prediction!'):
        with st.spinner('Working some magic...'):
            label, labels_percents, attentions, tokens = inference_transformer(
                model_pickle_path=model_tokenizer_pickle_path,
                text_input=user_input, ids_labels=IDS_LABELS)
            fig = plot_labels_confidence(labels_percentages=labels_percents, labels_coloring=LABELS_COLORS)
            html_text = html_highlight_text(weights=attentions, tokens=tokens, color=LABELS_COLORS[label],
                                            intensity=intensity)

        st.markdown(f'### **{label}**')

        st.markdown('### Confidence:')

        st.pyplot(fig)

        st.markdown('### **Text with attention color:**')
        st.markdown(html_text, unsafe_allow_html=True)
        
        try:
          del fig, html_text, label, labels_percents, attentions, tokens
        except:
          print('nothing to clean')
    gc.collect()

    return


def preconfigure_app(arguments):
    r"""
    Run preconfigure process for first time app run.
    """

    # Create config parser.
    config_file = configparser.ConfigParser()

    # Streamlit info
    st.subheader("This is the app's first run")
    st.write('Model will need to be downloaded and configured.')

    # Create folder if doesn't exists
    if not os.path.isdir(arguments.path_models):
        os.mkdir(arguments.path_models)

    # Create folder if doesn't exists
    if not os.path.isdir(arguments.model_tokenizer_pickle_path):
        os.mkdir(arguments.model_tokenizer_pickle_path)

    # Streamlit info
    st.write('Downloading models from `models_config.ini`...')

    # Download all pretrained models and updated config file
    download_from_config(arguments.path_model_config_file, arguments.path_models, use_streamlit=True)

    # Streamlit info
    st.write('Finished downloading models!')

    # Read config file from path.
    config_file.read(arguments.path_config_file)

    # Streamlit info
    st.write('\nPickle each model and tokenizer for fast inference:')

    # Parse each section in the config file.
    for section in config_file.sections():

        # Streamlit info
        st.write(f'  `{section}`')

        # Get the Google Drive download link
        model_path = config_file.get(section, 'model_path', fallback='')

        # Check if file actually exists
        if os.path.isdir(model_path):
            model_tokenizer_pickle_name = pickle_pytorch_models(model_path_=model_path,
                                                                pickled_path=arguments.model_tokenizer_pickle_path)
            # Add pickled model path to section.
            config_file.set(section, 'model_tokenizer_pickle_path', model_tokenizer_pickle_name)

    # Streamlit info
    st.write('Finished pickles!')

    # Writing updated configuration file with `model_path` added.
    with open(arguments.path_config_file, 'w') as configfile:
        config_file.write(configfile)

    # Streamlit info
    st.write('Updated configurations in `config.ini`!')
    st.subheader('App si ready to be used!\n')

    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    
    # Start with app header.
    app_header()

    # Parse any input arguments
    parser = argparse.ArgumentParser(description='Description')

    # Path of modified config file
    parser.add_argument('--path_model_config_file', help='Path where all pretrained models are stored pickled.',
                        type=str, default='models_config.ini')
    # Path of modified config file
    parser.add_argument('--path_config_file', help='Path where final config file containing all pickled models.',
                        type=str, default=CONFIG_FILE)

    # Path of pretrained downloaded models
    parser.add_argument('--path_models', help='Path where all pretrained models are stored pickled.',
                        type=str, default='pretrained_models')

    # Path of pickled models and tokenizers
    parser.add_argument('--model_tokenizer_pickle_path', help='Path where all pretrained models are stored pickled.',
                        type=str, default='pickled_models')

    # Parse arguments
    args = parser.parse_args()

    # Create config parser.
    config = configparser.ConfigParser()

    # Check if app preconfiguring was ran
    if not os.path.isfile(args.path_config_file):
        # If no config file is present - run preconfiguring
        preconfigure_app(arguments=args)

    # Read configuration file
    config.read(args.path_config_file)

    # Run modeling part of the app.
    app_modeling(config_file=config)
