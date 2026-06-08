%% Evaluate_PseudoOnline.m
% Official Pseudo-Online Evaluation Script for the PANCHO Framework
% Includes adjustable Out-of-Band Rejection (Idle State) thresholding.

clear; close all; clc;

%% 1. Configuration and Initialization
modelPath = fullfile('..', 'models', 'Pretrained_CNN_3x3_drop50.mat');
if ~exist(modelPath, 'file')
    error('Pretrained model not found. Please run Train_Lissajous_CNN.m first.');
end
load(modelPath); 
Net = CNN_model_test_v3;

sampleDataPath = fullfile('..', 'data', 'sample_EEG.mat');
if ~exist(sampleDataPath, 'file')
    error('Sample raw EEG data file not found.');
end
load(sampleDataPath); 

% Operational Parameters
Fs = fs;
WindowLength_sec = 0.5;
WindowSamples = round(WindowLength_sec * Fs); 
StepSamples = round(0.1 * Fs); % Slide every 100 ms
TimeVec = (0:WindowSamples-1)' / Fs;

% BCI Target Frequencies
Targets = [8.0, 13.0, 19.0, 25.0]; 
NumClasses = length(Targets);

% --- NUEVO: PARÁMETROS DE DIAGNÓSTICO Y RECHAZO ---
DecisionThreshold = 0.99; % 80% de confianza requerida para aceptar un Target
EnableDiagnostics = true; % Mostrar la primera ventana procesada
% --------------------------------------------------

ImSz = 128;
Lim = 1; 
NetDepth = 3; 
TgtIdx = 2; 

%% 2. Precompute References and Resonant Filters
fprintf('Precomputing resonant filters...\n');
RawOz = eeg_data(1, :)'; 
FilteredSignals = PrecomputeResonant(RawOz, Targets, Fs);
% FilteredSignals = PrecomputeResonant_BrickWall(RawOz, Targets, Fs);

RefSines = zeros(WindowSamples, NumClasses);
for k = 1:NumClasses
    RefSines(:, k) = sin(2 * pi * Targets(k) * TimeVec);
end

%% 3. Pseudo-Online Sliding Window Evaluation
fprintf('Starting PANCHO evaluation (Threshold: %.2f)...\n', DecisionThreshold);
TotalSamples = length(RawOz);
StartIndex = 1;
WindowCount = 0;

predict(Net, zeros(ImSz, ImSz, NetDepth, 'uint8'));

Latencies = [];
FinalCommands = [];
ConfidenceHistory = [];

while (StartIndex + WindowSamples - 1) <= TotalSamples
    WindowCount = WindowCount + 1;
    EndIndex = StartIndex + WindowSamples - 1;
    CurrentSignals = FilteredSignals(StartIndex:EndIndex, :);
    
    % tic;
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
    
    tic;
    % probs = predict(Net, BatchImages);
    % Obligamos a usar la CPU para evitar la latencia de transferencia al GPU en lotes pequeños
    probs = predict(Net, BatchImages, 'ExecutionEnvironment', 'cpu');

    InferenceTime = toc * 1000; % ms

    ProbsTgt = probs(:, TgtIdx); 
    
    % --- LÓGICA DE UMBRAL DE RECHAZO ---
    [MaxProb, PredIdx] = max(ProbsTgt);
    
    if MaxProb >= DecisionThreshold
        FinalCommands = [FinalCommands; Targets(PredIdx)];
    else
        FinalCommands = [FinalCommands; 0]; % 0 Hz = Idle / Non-Target
    end
    % -----------------------------------
    
    % InferenceTime = toc * 1000; 
    
    Latencies = [Latencies; InferenceTime];
    ConfidenceHistory = [ConfidenceHistory; MaxProb];
    
    % --- DIAGNÓSTICO VISUAL ---
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

%% 4. Reporting
assignin('base', 'FinalCommands', FinalCommands);
assignin('base', 'Latencies', Latencies);
assignin('base', 'ConfidenceHistory', ConfidenceHistory); % Para auditar las probabilidades

fprintf('\n--------------------------------------------------\n');
fprintf('Pseudo-Online Evaluation Summary:\n');
fprintf('Processed Windows: %d\n', WindowCount);
fprintf('Mean Algorithmic Latency: %.2f ms\n', mean(Latencies));
fprintf('Commands and Confidence stored in Workspace.\n');
fprintf('--------------------------------------------------\n');

%% Helper Functions (Iguales a la versión anterior)
function Signals = PrecomputeResonant(Raw, Targets, Fs)
    L = length(Raw); Signals = zeros(L, length(Targets));
    for k = 1:length(Targets)
        Fc = Targets(k); BW = 0.25;
        [b1, a1] = butter(2, [Fc-BW/2, Fc+BW/2]/(Fs/2), 'bandpass');
        [b2, a2] = butter(2, [2*Fc-BW/2, 2*Fc+BW/2]/(Fs/2), 'bandpass');
        Signals(:,k) = filtfilt(b1, a1, Raw) + 0.5 * filtfilt(b2, a2, Raw);
    end
end

% function Signals = PrecomputeResonant_BrickWall(Raw, Targets, Fs)
%     L = length(Raw); Signals = zeros(L, length(Targets));
%     for k = 1:length(Targets)
%         Fc = Targets(k); BW = 0.5;
% 
%         % ellip(Orden, Ripple_Passband_dB, Atten_Stopband_dB, Frecuencias)
%         % Garantiza 40 dB de atenuación justo fuera de la banda pasante
%         [b1, a1] = ellip(2, 0.5, 40, [Fc-BW/2, Fc+BW/2]/(Fs/2), 'bandpass');
%         [b2, a2] = ellip(2, 0.5, 40, [2*Fc-BW/2, 2*Fc+BW/2]/(Fs/2), 'bandpass');
% 
%         Signals(:,k) = filtfilt(b1, a1, Raw) + 0.5 * filtfilt(b2, a2, Raw);
%     end
% end

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