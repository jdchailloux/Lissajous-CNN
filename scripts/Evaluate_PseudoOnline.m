%% Evaluate_PseudoOnline.m
% =========================================================================
% PANCHO Framework: Pseudo-Online Evaluation Script
% =========================================================================
% Description:
%   Official pseudo-online evaluation script for the Lissajous-CNN BCI.
%   This script simulates a real-time environment by applying a sliding 
%   window, harmonic resonant filtering, 2D rasterization, and batch 
%   topological inference. It includes an out-of-band rejection threshold 
%   to handle biological SSVEP fatigue and idle states.
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

%% 1. Configuration and Initialization
% Load the pre-trained lightweight CNN
modelPath = fullfile('..', 'models', 'Pretrained_CNN_3x3_drop50.mat');
if ~exist(modelPath, 'file')
    error('Pretrained model not found in the /models/ directory.');
end
load(modelPath); 
Net = CNN_model_test_v3;

% Load sample raw EEG file
sampleDataPath = fullfile('..', 'data', 'sample_EEG.mat');
if ~exist(sampleDataPath, 'file')
    error('Sample raw EEG data file not found in the /data/ directory.');
end
load(sampleDataPath); 

% Operational Parameters
Fs = fs;
WindowLength_sec = 0.5;
WindowSamples = round(WindowLength_sec * Fs); 
StepSamples = round(0.1 * Fs); % Slide window every 100 ms
TimeVec = (0:WindowSamples-1)' / Fs;

% BCI Target Frequencies & Ground Truth
Targets = [11.0, 15.0, 20.0, 24.0]; 
NumClasses = length(Targets);
TrueTarget = 11.0; % <-- GROUND TRUTH LABEL of sample_EEG.mat (Adjust accordingly)

% Algorithm Parameters
DecisionThreshold = 0.0;  % 50% threshold recommended to accommodate natural SSVEP biological fatigue
EnableDiagnostics = true; % Set to true to visualize the first processed window
ImSz = 128; Lim = 1; NetDepth = 3; TgtIdx = 2; 

%% 2. Precompute References and Resonant Filters
fprintf('Precomputing resonant filters...\n');
RawOz = eeg_data(1, :)'; 
FilteredSignals = PrecomputeResonant(RawOz, Targets, Fs);

RefSines = zeros(WindowSamples, NumClasses);
for k = 1:NumClasses
    RefSines(:, k) = sin(2 * pi * Targets(k) * TimeVec);
end

%% 3. Pseudo-Online Sliding Window Evaluation
fprintf('Starting PANCHO evaluation (Threshold: %.2f)...\n', DecisionThreshold);
TotalSamples = length(RawOz);
StartIndex = 1;
WindowCount = 0;

% Warm-up CNN graph explicitly on CPU to prevent first-iteration overhead
predict(Net, zeros(ImSz, ImSz, NetDepth, 'uint8'), 'ExecutionEnvironment', 'cpu');

% Storage variables for workspace analysis
Latencies = [];
FinalCommands = [];
ConfidenceHistory = []; % Track maximum probabilities for confidence/fatigue analysis

while (StartIndex + WindowSamples - 1) <= TotalSamples
    WindowCount = WindowCount + 1;
    EndIndex = StartIndex + WindowSamples - 1;
    CurrentSignals = FilteredSignals(StartIndex:EndIndex, :);
    
    % --- PRE-PROCESSING (Excluded from inference latency metric) ---
    Energies = sum(CurrentSignals.^2, 1); 
    [~, WinSigIdx] = max(Energies);
    Seg_CNN = CurrentSignals(:, WinSigIdx); 
    
    if max(abs(Seg_CNN)) > 0
        Seg_CNN = Seg_CNN / max(abs(Seg_CNN)); 
    end
    
    BatchImages = zeros(ImSz, ImSz, NetDepth, NumClasses, 'uint8'); 
    
    for k = 1:NumClasses
        Ref = RefSines(:, k); 
        GrayImg = FastLissajousRaster(Ref, Seg_CNN, ImSz, Lim);
        if NetDepth == 3
            BatchImages(:,:,:,k) = repmat(GrayImg, [1 1 3]);
        else
            BatchImages(:,:,1,k) = GrayImg; 
        end
    end
    % ----------------------------------------------------------------
    
    % --- STRICT CNN INFERENCE LATENCY MEASUREMENT ---
    tic;
    probs = predict(Net, BatchImages, 'ExecutionEnvironment', 'cpu'); 
    InferenceTime = toc * 1000; 
    % --------------------------------------------------------
    
    ProbsTgt = probs(:, TgtIdx); 
    [MaxProb, PredIdx] = max(ProbsTgt);
    
    % Apply Out-of-Band Rejection Threshold
    if MaxProb >= DecisionThreshold
        FinalCommands = [FinalCommands; Targets(PredIdx)];
    else
        FinalCommands = [FinalCommands; 0]; % 0 Hz = Idle state / Noise
    end
    
    % Store iteration metrics
    Latencies = [Latencies; InferenceTime];
    ConfidenceHistory = [ConfidenceHistory; MaxProb]; 
    
    % --- Visual Diagnostics ---
    if EnableDiagnostics && WindowCount == 1
        figure('Name', 'Lissajous Diagnostic (Window 1)', 'Color', 'w');
        tiledlayout(1, 2, 'Padding', 'compact');
        
        nexttile;
        plot(TimeVec, Seg_CNN, 'b', 'LineWidth', 1.5); hold on;
        plot(TimeVec, RefSines(:, PredIdx), 'r--', 'LineWidth', 1.2);
        title(sprintf('Time Domain (Winner Energy: %.1f Hz)', Targets(PredIdx)));
        legend('EEG Y-Axis', 'Template X-Axis'); xlabel('Time (s)'); xlim([0 WindowLength_sec]);
        
        nexttile;
        imshow(BatchImages(:,:,:,PredIdx));
        if MaxProb >= DecisionThreshold
            title(sprintf('ACCEPTED Target (Conf: %.1f%%)', MaxProb*100));
        else
            title(sprintf('REJECTED as Noise (Conf: %.1f%%)', MaxProb*100), 'Color', 'r');
        end
        drawnow;
    end
    
    StartIndex = StartIndex + StepSamples;
end

%% 4. Reporting & Metrics Calculation
% Export variables to MATLAB Base Workspace for user inspection
assignin('base', 'FinalCommands', FinalCommands);
assignin('base', 'Latencies', Latencies);
assignin('base', 'ConfidenceHistory', ConfidenceHistory); 

% Generate True Labels vector for the entire evaluation
TrueLabelsVec = repmat(TrueTarget, length(FinalCommands), 1);

% Compute Metrics internally
[Accuracy, ITR, Kappa, ~] = Compute_Metrics(TrueLabelsVec, FinalCommands, WindowLength_sec, NumClasses);

fprintf('\n--------------------------------------------------\n');
fprintf('Pseudo-Online Evaluation Summary:\n');
fprintf('Processed Windows: %d\n', WindowCount);
fprintf('Mean CNN Latency:  %.2f ms\n', mean(Latencies));
fprintf('Commands and Confidence scores stored in Workspace.\n');
fprintf('--------------------------------------------------\n');
fprintf('Performance Metrics:\n');
fprintf('Accuracy:          %.2f%%\n', Accuracy * 100);
fprintf('ITR:               %.2f bits/min\n', ITR);

% Visual safeguard for Cohen's Kappa in single-trial evaluations
if length(unique(TrueLabelsVec)) == 1
    fprintf('Kappa (k):         N/A (Multi-class data required)\n');
else
    fprintf('Kappa (k):         %.4f\n', Kappa);
end
fprintf('--------------------------------------------------\n');

%% =========================================================================
%                          HELPER FUNCTIONS
% =========================================================================

function [Accuracy, ITR, Kappa, ConfMat] = Compute_Metrics(TrueLabels, PredictedLabels, WindowLength, NumClasses)
    TotalTrials = length(TrueLabels);
    Po = sum(TrueLabels(:) == PredictedLabels(:)) / TotalTrials;
    Accuracy = Po;
    
    P_itr = max(1/NumClasses, min(Po, 0.9999));
    B = log2(NumClasses) + P_itr * log2(P_itr) + (1 - P_itr) * log2((1 - P_itr) / (NumClasses - 1));
    ITR = B * (60 / WindowLength);
    
    AllLabels = unique([TrueLabels; PredictedLabels]); 
    ConfMat = confusionmat(TrueLabels, PredictedLabels, 'Order', AllLabels);
    n = sum(ConfMat(:));
    Pe = sum(sum(ConfMat, 2) .* sum(ConfMat, 1)') / (n^2);
    
    if Pe == 1
        Kappa = 0; 
    else
        Kappa = (Po - Pe) / (1 - Pe); 
    end
end

function Signals = PrecomputeResonant(Raw, Targets, Fs)
    L = length(Raw); Signals = zeros(L, length(Targets));
    for k = 1:length(Targets)
        Fc = Targets(k); BW = 0.25;
        [b1, a1] = butter(2, [Fc-BW/2, Fc+BW/2]/(Fs/2), 'bandpass');
        [b2, a2] = butter(2, [2*Fc-BW/2, 2*Fc+BW/2]/(Fs/2), 'bandpass');
        Signals(:,k) = filtfilt(b1, a1, Raw) + 0.5 * filtfilt(b2, a2, Raw);
    end
end

function Img = FastLissajousRaster(X, Y, Sz, Lim)
    Xi = round( ((X + Lim) / (2*Lim)) * (Sz-1) ) + 1; Yi = round( ((Y + Lim) / (2*Lim)) * (Sz-1) ) + 1;
    Xi(Xi < 1) = 1; Xi(Xi > Sz) = Sz; Yi(Yi < 1) = 1; Yi(Yi > Sz) = Sz;
    
    NumPoints = length(X); UpsampleFactor = 4;
    Xi_up = interp1(1:NumPoints, Xi, linspace(1, NumPoints, NumPoints*UpsampleFactor));
    Yi_up = interp1(1:NumPoints, Yi, linspace(1, NumPoints, NumPoints*UpsampleFactor));
    
    IndX = round(Xi_up); IndY = round(Yi_up);
    Rows = Sz - IndY + 1; Cols = IndX; 
    LinInds = sub2ind([Sz, Sz], Rows, Cols);
    
    Img = true(Sz, Sz); Img(LinInds) = false; 
    SE = [0 1 0; 1 1 1; 0 1 0]; Img = imerode(Img, SE); Img = uint8(Img) * 255;
end