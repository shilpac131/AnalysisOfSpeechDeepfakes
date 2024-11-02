"""
Main script that evaluates fake or real audios using AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

'''the docker doesn't recongnize paths from system

so

docker run --gpus all -it -v /path/to/your/audio/folder:/container/audio/path <docker-image-name>
example: docker run --gpus all -it -v ./mydockerdata:/app/data deepfake_detector
'''

import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union
from torch import Tensor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA
import numpy as np
from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list)
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool
import soundfile as sf

warnings.filterwarnings("ignore", category=FutureWarning)

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """

    # making changes for user input 
    print("THIS IS THE AASIST audio deepfake algorithm - FOR EVALUATION ONLY")
    choice = input("enter \n1. For a Single Audio \n2. For an entire audio folder path\n")
    print("\n You choose : ",choice)

    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)


    # evaluates pretrained model and exit script
    if args.eval:
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")

        # eval according to the choices by the user

        if choice == '1':
            audio_path = input("enter audio path  \n->")

            print('Audio loaded : {}'.format(audio_path))
            X,fs = sf.read(audio_path) 
            X_pad= pad(X,64600)
            x_inp= Tensor(X_pad)
            x_inp = x_inp.view(1, -1)
            print("audio file loaded")

            print("let's begin inference")
            model.eval()
            x_inp = x_inp.to(device)
            _,pred = model(x_inp)
            softmax_probs = torch.softmax(pred, dim=1)
            _, predicted_class = torch.max(softmax_probs, 1)
            # Get the softmax probability values of the predicted class
            predicted_class_probs = softmax_probs.gather(1, predicted_class.unsqueeze(1)).squeeze(1)
            if predicted_class.item() == 0:
                print(f"The audio is spoofed with confidence score of {predicted_class_probs.item()}")
            elif predicted_class.item() == 1:
                print(f"The audio is bonafide with confidence score of {predicted_class_probs.item()}")
        elif choice == '2':

            folder_path = input("enter audio folder path  \n->")
            print('Audio loaded : {}'.format(folder_path))
            audio_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.wav', '.flac'))] 
            model.eval()
            spoof = 0
            count_spoof = 0
            bonafide = 0
            count_bonafide = 0

            ## FOR DOCKER:
            output_file = '/app/data/output.txt'
            with open(output_file, "a") as file:
                for audio_file in audio_files:
                    X,fs = sf.read(audio_file) 
                    X_pad= pad(X,64600)
                    x_inp= Tensor(X_pad)
                    x_inp = x_inp.view(1, -1)
                    x_inp = x_inp.to(device)
                    _,pred = model(x_inp)
                    softmax_probs = torch.softmax(pred, dim=1)
                    _, predicted_class = torch.max(softmax_probs, 1)
                    # Get the softmax probability values of the predicted class
                    predicted_class_probs = softmax_probs.gather(1, predicted_class.unsqueeze(1)).squeeze(1)
                    if predicted_class.item() == 0:
                        count_spoof = count_spoof+1
                        file.write(f"\nPrediction of File {audio_file} is:> spoofed with confidence score of {predicted_class_probs.item()}")
                    elif predicted_class.item() == 1:
                        count_bonafide = count_bonafide+1
                        file.write(f"\nPrediction of File {audio_file} is:> bonafide with confidence score of {predicted_class_probs.item()}")
            print(f"Predictions saved in path-> ./mydockerdata")
            print("Total spoof:> ",count_spoof)
            print("Total bonafide:> ",count_bonafide)
        sys.exit(0)


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        default = "./config/AASIST.conf",
                        type=str,
                        help="configuration file",
                        required=False)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())
