import os
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import numpy as np
import librosa

from .dswipe import dSWIPE
from .dyin import dYIN


def predict(model, fn_audio):
    x = librosa.load(fn_audio, sr=model.fs)[0]
    x = torch.from_numpy(x)

    out = model(x)

    frequency = out["f0_hz"].cpu().detach().numpy()
    time = np.arange(frequency.size) * model.hop_size / model.fs
    return time, frequency


def process_files(model, audio_files, dir_out):
    if isinstance(audio_files, str):
        audio_files = [audio_files]

    os.makedirs(dir_out, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for file in tqdm(audio_files):
            time, frequency = predict(model, file)

            fn = Path(file).stem
            fn_out = os.path.join(dir_out, f"{fn}.csv")

            data = np.stack((time, frequency), axis=1)
            header = "time,frequency"
            np.savetxt(fn_out, data, delimiter=",", fmt="%.3f", header=header, comments="")


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "audio_files", metavar="FILE", type=str, nargs="+", help="Audio files to process"
    )
    parser.add_argument(
        "--dir_out",
        metavar="DIR",
        type=str,
        default="f0_csv",
        help="Directory to save the output predictions",
    )
    parser.add_argument("--fs", type=int, default=16000, help="Sampling frequency in Hz")
    parser.add_argument("--hop_size", type=int, default=320, help="Hop size in samples")
    parser.add_argument("--f0_min", type=float, default=55.0, help="Lowest detectable F0 in Hz")
    parser.add_argument("--f0_max", type=float, default=3520.0, help="Highest detectable F0 in Hz")
    parser.add_argument(
        "--f0_r_cent", type=float, default=10.0, help="Output frequency resolution in cents"
    )
    parser.add_argument(
        "--f0_selection_strategy",
        type=str,
        default="argmax",
        choices=["argmax", "parabolic_interpolation", "local_weighted_average"],
        help="Specifies the F0 selection strategy",
    )
    return parser


def dswipe():
    parser = get_parser()
    parser.add_argument(
        "--erb_f_min",
        type=float,
        default=13.75,
        help="Minimum frequency of the ERB-based frequency axis in Hz",
    )
    parser.add_argument(
        "--erb_f_max",
        type=float,
        default=8000.0,
        help="Maximum frequency of the ERB-based frequency axis in Hz",
    )
    parser.add_argument(
        "--erb_r",
        type=float,
        default=0.1,
        help="Resolution of the ERB-based frequency axis in ERB units",
    )
    args = parser.parse_args()

    dswipe_config = {
        "fs": args.fs,
        "hop_size": args.hop_size,
        "erb_f_min": args.erb_f_min,
        "erb_f_max": args.erb_f_max,
        "erb_r": args.erb_r,
        "f0_min": args.f0_min,
        "f0_max": args.f0_max,
        "f0_r_cent": args.f0_r_cent,
        "f0_selection_strategy": args.f0_selection_strategy,
    }

    dswipe = dSWIPE(**dswipe_config)
    process_files(dswipe, args.audio_files, args.dir_out)


def dyin():
    parser = get_parser()
    parser.add_argument("--frame_size", type=int, default=1600, help="Frame size in samples")
    args = parser.parse_args()

    dyin_config = {
        "fs": args.fs,
        "frame_size": args.frame_size,
        "hop_size": args.hop_size,
        "f0_min": args.f0_min,
        "f0_max": args.f0_max,
        "f0_r_cent": args.f0_r_cent,
        "f0_selection_strategy": args.f0_selection_strategy,
    }

    dyin = dYIN(**dyin_config)
    process_files(dyin, args.audio_files, args.dir_out)
