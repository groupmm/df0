# df0: Differentiable Variants of Classical Fundamental Frequency Estimators

This is a Python package containing Pytorch implementations of differentiable variants of classical fundamental frequency estimators (dYIN and dSWIPE). This code accompanies the following paper:

```bibtex
@article{StrahlM25_dYIN_dSWIPE,
  author    = {Sebastian Strahl and Meinard M{\"u}ller},
  title     = {{dYIN} and {dSWIPE}: {D}ifferentiable Variants of Classical Fundamental Frequency Estimators},
  journal   = {},
  volume    = {},
  pages     = {},
  year      = {},
}
```

For details and references, please check out this paper.


## Installation

### 1. Set up Python environment
We recommend setting up a Python environment including Pytorch before installing `df0`. You may use the example environment provided as part of this package:

```bash
git clone https://github.com/groupmm/df0.git
cd df0
conda env create -f environment.yaml
```

### 2. Install `df0`

#### Option 1: Installation without cloning this repository:

```bash
pip install "git+https://github.com/groupmm/df0.git#egg=df0"
```

#### Option 2: Installation by cloning this repository:

```bash
git clone https://github.com/groupmm/df0.git
cd df0
pip install -e .
```

#### Option 3: Install all packages to run the demo notebook:

```bash
pip install "git+https://github.com/groupmm/df0.git#egg=df0[demo]"
```

or

```bash
git clone https://github.com/groupmm/df0.git
cd df0
pip install -e .[demo]
```

## Usage

### Python

The usage of dYIN and dSWIPE as a differentiable Pytorch module is demonstrated in [demo.ipynb](demo.ipynb).


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


## Contribution
Automated code style checks via [pre-commit](https://pre-commit.com/):

```bash
pip install pre-commit
pre-commit install
```


## License
The code for this toolbox is published under an [MIT license](LICENSE).
This does not apply to the data files, which are taken from the [FMP notebooks](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C0/C0.html).


## Acknowledgements

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Grant No. 500643750 (MU 2686/15-1). The authors are with the [International Audio Laboratories Erlangen](https://audiolabs-erlangen.de/), a joint institution of the [Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)](https://www.fau.eu/) and [Fraunhofer Institute for
Integrated Circuits IIS](https://www.iis.fraunhofer.de/en.html).
