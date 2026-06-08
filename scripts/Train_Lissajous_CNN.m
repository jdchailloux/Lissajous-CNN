%% Train_Lissajous_CNN.m
% Official training script for the PANCHO Framework (Lissajous-CNN)
% This script builds and trains the lightweight CNN using synthetic data.

clear; close all; clc;

%% 1. Load Synthetic Dataset Path
% Define paths for the synthetically generated Lissajous image dataset
datasetPath = fullfile('..', 'data', 'synthetic_images');

% Ensure the dataset exists before proceeding
if ~exist(datasetPath, 'dir')
    error('Synthetic dataset directory not found. Please run image generation first.');
end

% Create an Image Datastore automatically labeling images based on folder names
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split data into Training (80%) and Validation (20%) sets
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

%% 2. Define the Optimized Lightweight CNN Architecture
% This configuration yields exactly 34,210 learnable parameters
layers = [
    imageInputLayer([128 128 3], 'Name', 'Input')
    
    % First Convolutional Block
    convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'Conv1')
    batchNormalizationLayer('Name', 'BatchNorm1')
    reluLayer('Name', 'ReLU1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPool1') % Down to 64x64x8
    
    % Second Convolutional Block
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'Conv2')
    batchNormalizationLayer('Name', 'BatchNorm2')
    reluLayer('Name', 'ReLU2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPool2') % Down to 32x32x16
    
    % Regularization and Classification Layer
    dropoutLayer(0.5, 'Name', 'Dropout_50')
    fullyConnectedLayer(2, 'Name', 'FC_2Classes') % Target vs Non-Target
    softmaxLayer('Name', 'Softmax')
    classificationLayer('Name', 'Output')
];

%% 3. Specify Training Options
% Optimized hyperparameters using the Adam solver for fast convergence
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% 4. Train the Model
fprintf('Starting training for Lissajous-CNN Model...\n');
[net, info] = trainNetwork(imdsTrain, layers, options);

%% 5. Save the Trained Model
outputFolder = fullfile('..', 'models');
if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end

modelPath = fullfile(outputFolder, 'Pretrained_CNN_3x3_drop50.mat');
CNN_model_test_v3 = net; % Match manuscript variable naming convention
save(modelPath, 'CNN_model_test_v3');
fprintf('Model successfully trained and saved to: %s\n', modelPath);