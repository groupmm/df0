# df0: Differentiable Variants of Classical Fundamental Frequency Estimators

[![PyPI version](https://img.shields.io/pypi/v/df0-pitch.svg)](https://pypi.org/project/df0-pitch/)
[![tests](https://github.com/groupmm/df0/actions/workflows/tests.yml/badge.svg)](https://github.com/groupmm/df0/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This is a Python package containing Pytorch implementations of differentiable variants of classical fundamental frequency estimators (dYIN and dSWIPE). This code accompanies the following paper:

```bibtex
@article{StrahlM25_df0_TASLPRO,
  author  = {Sebastian Strahl and Meinard M{\"u}ller},
  title   = {{dYIN} and {dSWIPE}: {D}ifferentiable Variants of Classical Fundamental Frequency Estimators},
  journal = {{IEEE/ACM} Transactions on Audio, Speech, and Language Processing},
  volume  = {33},
  pages   = {2622--2633},
  year    = {2025},
  doi     = {10.1109/TASLPRO.2025.3581119}
}
```

For details and references, please check out this paper.


## Installation

### 1. Set up Python environment
We recommend setting up a Python environment including Pytorch before installing `df0`. You may use the [example environment](environment.yaml) provided as part of this package:

```bash
git clone https://github.com/groupmm/df0.git
cd df0
conda env create -f environment.yaml
conda activate df0
```

### 2. Install `df0`

#### Option 1: Install from PyPI

```bash
pip install df0-pitch
```

To also install the dependencies required to run the demo notebook:

```bash
pip install "df0-pitch[demo]"
```

#### Option 2: Install by cloning this repository (for development)

```bash
git clone https://github.com/groupmm/df0.git
cd df0
pip install -e .
```

## Usage

### Python

```python
import torch
import librosa
from df0.dyin import dYIN
from df0.dswipe import dSWIPE

x, fs = librosa.load("audio.wav", sr=16000)
x = torch.from_numpy(x)

dyin = dYIN(fs=fs, hop_size=160)
f0_hz = dyin(x)["f0_hz"]  # shape: (n_frames,)

dswipe = dSWIPE(fs=fs, hop_size=160)
f0_hz = dswipe(x)["f0_hz"]  # shape: (n_frames,)
```

For more details, see [demo.ipynb](demo.ipynb).


### Command-line interface

To get the F0 predictions for an audio file, you can use the command-line interface, running one of the following commands in a terminal:

```bash
dswipe audio.wav
dyin audio.wav
```

To get the predictions for all `.wav` files in `dir_audio/`, use:

```bash
dswipe dir_audio/*.wav
```

By running these commands, the F0 predictions are stored as `.csv` files in a folder specified by the command-line argument `dir_out` (default: `f0_csv`). Example usage:

```bash
dswipe dir_audio/*.wav --dir_out f0_dswipe_csv
```

Use `dswipe -h` or `dyin -h` for an overview of available command-line options.

The output format of our methods is similar to that of [CREPE](https://github.com/marl/crepe) and [PESTO](https://github.com/SonyCSLParis/pesto/tree/master), except that there is no confidence:
```csv
time,frequency
0.000,63.544
0.010,651.683
0.020,655.458
0.030,651.683
0.040,666.915
0.050,678.573
...
```

## Tests

We provide automated tests for each algorithm. To run them:

```bash
pip install "df0[test]"
pytest
```

## Contribution
Automated code style checks via [pre-commit](https://pre-commit.com/):

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```


## License
The code for this toolbox is published under an [MIT license](LICENSE).


## Acknowledgements

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Grant No. 500643750 (MU 2686/15-1). The authors are with the [International Audio Laboratories Erlangen](https://audiolabs-erlangen.de/), a joint institution of the [Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)](https://www.fau.eu/) and [Fraunhofer Institute for Integrated Circuits IIS](https://www.iis.fraunhofer.de/en.html).
