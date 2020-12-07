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
"""Run this script to load model and pickle them for faster inference."""


import pickle
import shutil
import os
import sys
import argparse
import configparser
from transformers import (AutoConfig,
                          AutoModelForSequenceClassification,
                          AutoTokenizer,
                          set_seed)
from settings import CONFIG_FILE



def pickle_pytorch_models(model_path_, pickled_path):
    print(f'Loading configuration, tokenizer and model from: `{model_path_}`')
    sys.stdout.flush()
    # Set seed for reproducibility,
    set_seed(123)

    # Get model configuration.
    print('Loading configuration...')
    sys.stdout.flush()
    model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_path_,
                                              )

    # Get model's tokenizer.
    print('Loading tokenizer...')
    sys.stdout.flush()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path_)

    # Get the actual model.
    print('Loading model...')
    sys.stdout.flush()
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path_,
                                                               config=model_config)

    model_tokenizer_pickle_name_ = os.path.join(pickled_path, f'{os.path.basename(model_path_)}.pickle')

    with open(model_tokenizer_pickle_name_, 'wb') as handle:
        pickle.dump([tokenizer, model], handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Model and Tokenizer pickled at:                  `{model_tokenizer_pickle_name_}`\n')
    sys.stdout.flush()

    # Remove pretrained
    try:
        shutil.rmtree(model_path_)
    except OSError as e:
        print("Error: %s : %s" % (model_path_, e.strerror))
        sys.stdout.flush()

    return model_tokenizer_pickle_name_


# Main run of the script.
if __name__ == '__main__':

    # Parse any input arguments
    parser = argparse.ArgumentParser(description='Description')

    # Path of pickled models and tokenizers
    parser.add_argument('--model_tokenizer_pickle_path', help='Path where all pretrained models are stored pickled.',
                        type=str, default='pickled_models')

    # Parse arguments
    args = parser.parse_args()

    # Create folder if doesn't exists
    os.mkdir(args.model_tokenizer_pickle_path) if not os.path.isdir(args.model_tokenizer_pickle_path) else None

    # Create config parser.
    config = configparser.ConfigParser()

    # Read config file from path.
    config.read(CONFIG_FILE)

    # Parse each section in the config file.
    for section in config.sections():
        # Get the Google Drive download link
        model_path = config.get(section, 'model_path', fallback='')

        # Check if file actually exists
        if os.path.isdir(model_path):
            model_tokenizer_pickle_name = pickle_pytorch_models(model_path_=model_path,
                                                                pickled_path=args.model_tokenizer_pickle_path)
            # Add pickled model path to section.
            config.set(section, 'model_tokenizer_pickle_path', model_tokenizer_pickle_name)

    # Writing updated configuration file with `model_path` added.
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)

    print(f'\nFinished running `{__file__}`!')
    sys.stdout.flush()
