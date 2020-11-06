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
"""Download files and read configuration file"""

import configparser
import streamlit as st
import gdown
import os
import zipfile
import argparse
from settings import CONFIG_FILE


def download_from_config(path_config_file, path_downloaded_models, use_streamlit=False):
    r"""
    Download all pretrained models using config file and updating the config file with each model's path.

    Arguments:

        path_config_file:

        path_downloaded_models:
            this is where we download all models


    """

    # Create config parser.
    config = configparser.ConfigParser()

    # Read config file from path.
    config.read(path_config_file)

    # Parse each section in the config file.
    for section in config.sections():

        # Streamlit info
        if use_streamlit:
            st.write(f'  `{section}`')

        # Get the Google Drive download link
        download_link = config.get(section, 'google_drive_zip_link', fallback=None)

        # Get file name with path to .zip - where we will save the archive.
        file_name = os.path.join(path_downloaded_models, f'{section}.zip')

        # Download the .zip from Google Drive.
        gdown.download(url=download_link, output=file_name, quiet=False)

        # Check if file actually exists
        if os.path.isfile(file_name):

            # File .zip exists. Let's unzip it
            with zipfile.ZipFile(file_name, "r") as zip_ref:

                # Get info of zip.
                zip_info = zip_ref.infolist()

                # Get folder name if first is a folder.
                folder_name = zip_info[0].filename if zip_info[0].is_dir() else None

                # Normalize folder name
                folder_name = os.path.normpath(folder_name)

                # Extract all zip components.
                zip_ref.extractall(path=path_downloaded_models)

            # Find out name of new folder
            if folder_name is None:

                # .zip does not have a main folder. Skip this section of config file.
                print('Zip does not have folder!')
                continue

            # Add model path to section.
            config.set(section, 'model_path', os.path.join(path_downloaded_models, folder_name))

            # Print info of model path.
            print(f'Unzip to: {os.path.join(path_downloaded_models, folder_name)}')

            # Streamlit info
            if use_streamlit:
                st.write(f'  Unzip to: `{os.path.join(path_downloaded_models, folder_name)}`')

            # Remove .zip to keep memory clean
            try:
                os.remove(file_name)
            except OSError as e:
                print("Error: %s : %s" % (file_name, e.strerror))

        # Separation line.
        print('--------------------------------------------------------------------------------------')

    # Writing updated configuration file with `model_path` added.
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)

    return


# Main run of the script.
if __name__ == '__main__':

    # Parse any input arguments
    parser = argparse.ArgumentParser(description='Description')

    # Path of modified config file
    parser.add_argument('--path_config_file', help='Path where all pretrained models are stored pickled.',
                        type=str, default='models_config.ini')

    # Path of pretrained downloaded models
    parser.add_argument('--path_models', help='Path where all pretrained models are stored pickled.',
                        type=str, default='pretrained_models')

    # Parse arguments
    args = parser.parse_args()

    # Create folder if doesn't exists
    os.mkdir(args.path_models) if not os.path.isdir(args.path_models) else None

    # Download all pretrained models and updated config file
    download_from_config(args.path_config_file, args.path_models)

    print(f'\nFinished running `{__file__}`!')
