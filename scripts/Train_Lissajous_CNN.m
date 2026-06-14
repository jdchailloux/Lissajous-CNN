%% Train_Lissajous_CNN.m
% =========================================================================
% PANCHO Framework: Lissajous-CNN Training Script
% =========================================================================
% Description:
%   This script builds, trains, and validates the lightweight 2D Convolutional 
%   Neural Network (Lissajous-CNN) using synthetic phase-space images.
%   This specific architecture corresponds to the optimal configuration 
%   (3x3 kernels, 50% Dropout) reported in the ablation study.
%
% Author:
%   Juan David Chailloux-Peguero
%   Tecnologico de Monterrey, Guadalajara Campus
%   (with the assistance of Gemini LLM)
%
% Reference:
%   Submitted to Biomedical Signal Processing and Control (Elsevier)
% =========================================================================

clear; close all; clc;

%% 1. Load Synthetic Dataset
fprintf('--- Initializing Lissajous-CNN Training Pipeline ---\n');

% Define the relative path for the synthetically generated dataset
datasetPath = fullfile('..', 'data', 'synthetic_images');

% Ensure the dataset directory exists before proceeding
if ~exist(datasetPath, 'dir')
    error('Synthetic dataset not found. Please run the image generation script first.');
end

% Create an Image Datastore, automatically labeling images based on subfolder names
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Set random seed for scientific reproducibility
rng(42); 

% Split data: 80% for Training, 20% for Validation
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');
fprintf('Dataset loaded successfully. Training samples: %d, Validation samples: %d\n', ...
    numel(imdsTrain.Files), numel(imdsValidation.Files));

%% 2. Define the Optimized Lightweight Architecture
% This topological configuration yields exactly 34,210 learnable parameters
layers = [
    imageInputLayer([128 128 3], 'Name', 'Input')
    
    % First Convolutional Block
    convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'Conv1_3x3')
    batchNormalizationLayer('Name', 'BatchNorm1')
    reluLayer('Name', 'ReLU1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPool1') % Downsamples to 64x64x8
    
    % Second Convolutional Block
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'Conv2_3x3')
    batchNormalizationLayer('Name', 'BatchNorm2')
    reluLayer('Name', 'ReLU2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPool2') % Downsamples to 32x32x16
    
    % Regularization and Classification Block
    dropoutLayer(0.5, 'Name', 'Dropout_50')
    fullyConnectedLayer(2, 'Name', 'FC_2Classes')         % Binary classification (Target vs. Non-Target)
    softmaxLayer('Name', 'Softmax')
    classificationLayer('Name', 'Output')
];

%% 3. Specify Training Hyperparameters
% Optimized options using the Adam optimizer for fast convergence
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% 4. Execute Network Training
fprintf('Commencing model training...\n');
[net, info] = trainNetwork(imdsTrain, layers, options);

%% 5. Export and Save the Trained Model
outputFolder = fullfile('..', 'models');
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder); 
end

% Save the model matching the manuscript's variable naming convention
modelPath = fullfile(outputFolder, 'Pretrained_CNN_3x3_drop50.mat');
CNN_model_test_v3 = net; 
save(modelPath, 'CNN_model_test_v3');

fprintf('\n--- Training Complete ---\n');
fprintf('Model successfully saved to: %s\n', modelPath);