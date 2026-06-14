# Lissajous-CNN (The PANCHO Framework) for SSVEP Decoding

[![MATLAB](https://img.shields.io/badge/MATLAB-R2023a%2B-blue.svg)](https://www.mathworks.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository contains the official MATLAB implementation of the **PANCHO framework** (Pattern analysis, Artificial intelligence, Normalization, Calibration-free, and Hardware optimization), a novel single-channel Brain-Computer Interface (BCI) decoding algorithm for Steady-State Visual Evoked Potentials (SSVEP).

Instead of relying on multi-channel spatial filtering and exhaustive human calibration, this method projects 1D EEG time-series into 2D topological invariants (Lissajous curves). A lightweight Convolutional Neural Network (CNN) trained exclusively on synthetically generated data is then used to classify the geometric morphology of the phase-space trajectory, achieving high-speed ($\le 1.0$ s), zero-shot decoding.

## Video demonstration of the project
[![YouTube Video Preview]
(https://img.youtube.com/vi/N8jD6kLlnsI/maxresdefault.jpg)]
(https://www.youtube.com/watch?v=N8jD6kLlnsI)


## Repository Structure
* `/data/`: Contains a sample pre-processed EEG trial (`sample_EEG.mat`) for testing.
* `/models/`: Contains the pre-trained `Pretrained_CNN_3x3_drop50.mat` zero-shot model.
* `/scripts/`:
  * `Train_Lissajous_CNN.m`: Script to construct and train the CNN architecture using synthetic data.
  * `Evaluate_PseudoOnline.m`: All-in-one script to benchmark the trained CNN against the sample EEG data, including resonant filtering, rasterization, and automated metrics computation.

## Prerequisites
To run the code, you will need MATLAB installed with the following toolboxes:
* Deep Learning Toolbox
* Signal Processing Toolbox
* Statistics and Machine Learning Toolbox

## Getting Started
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/Lissajous-CNN.git](https://github.com/your-username/Lissajous-CNN.git)

2. **Run Inference on Sample Data:**
Open MATLAB, navigate to the /scripts/ folder, and execute `Evaluate_PseudoOnline.m` to see the zero-shot classification in action over a 0.5 s observation window.

## Citation
If you find this code useful in your research, please cite our paper (NOW UNDER REVIEW):

## License
This project is licensed under the MIT License - see the LICENSE file for details.
