% =========================================================================
% Elin State Tagging – Training / Validation Script (single-file version)
% =========================================================================
% This script trains or verifies a CNN on STFT-based features created from a
% recorded signal. It supports iterative (non-overlapping) STFT sampling,
% label correction, class undersampling (two strategies), and evaluation.
%
% Notes:
% - Assumes helper functions exist on MATLAB path:
%   initializeVariables, getStftParameters, getUserChoice, getUserInput,
%   shortenData, map_to_stft_domain, prepareDeepLearningData,
%   applyFirstUndersamplingStrategy, applySecondUndersamplingStrategy
%
% Author: Adnan S.
% ========================================================================

close all; clearvars; clc;
rng('default');  % (optional) reproducible randsample

%% ------------------------------------------------------------------------
%  Initialize containers and STFT parameters
% -------------------------------------------------------------------------
[finalUndersampledX, finalUndersampledY, finalUndersampledValX, finalUndersampledValY] = initializeVariables();
[windowLength, fftLength, hopSize, overlapLength, numberOfWindows, win] = getStftParameters();
fs = 34133;  % sampling rate used in the original melSpectrogram call

%% ------------------------------------------------------------------------
%  User inputs (strategy, iterations, split, action, label-correction)
% -------------------------------------------------------------------------
strategies = {'No Undersampling', 'Undersampling Strategy 1', 'Undersampling Strategy 2'};
undersampling_strategy_values = [0, 1, 2];
UNDERSAMPLE_TRAINING_DATA = getUserChoice( ...
    strategies, undersampling_strategy_values, ...
    'Select an undersampling strategy:', ...
    'No strategy selected, defaulting to No Undersampling.');

total_number_of_iterations = getUserInput( ...
    'Enter total number of iterations (must be divisible by 2 and not exceed windowLength):', ...
    'Input for Iterations', '4', @(x) mod(x, 2)==0 && x <= windowLength);

splitRatio = getUserInput( ...
    'Enter the training/validation split ratio (between 0.1 and 0.9):', ...
    'Input for Training/Validation Split Ratio', '0.5', @(x) x >= 0.1 && x <= 0.9);

optionsAct = {'Train Model', 'Verify Model'};
option_values = [1, 2];  % TRAIN_MODEL = 1, VERIFY_MODEL = 2
train_or_verify_model = getUserChoice( ...
    optionsAct, option_values, ...
    'Select an action:', ...
    'No option selected, defaulting to Train Model.');

correct_misslabeled_data = getUserChoice( ...
    {'No', 'Yes'}, [0, 1], ...
    'Correct mislabeled data?', ...
    'No option selected, defaulting to No.');

shorten_data_to_one_third = 1;

%% ------------------------------------------------------------------------
%  Iterative dataset construction (non-overlapping STFT positions)
% -------------------------------------------------------------------------
% Resolve project root relative to this script:  src/ -> (up) -> project root
projectRoot = fileparts(fileparts(mfilename('fullpath')));

% Point to the .mat inside ../data/
dataMatFile = fullfile(projectRoot, 'data', 'segmented_data_elin_48kHz_pos1-27.mat');

if ~isfile(dataMatFile)
    error('Data file not found: %s', dataMatFile);
end

for iteration_step = 1:total_number_of_iterations
    fprintf('\n=== Iteration %d / %d ===\n', iteration_step, total_number_of_iterations);

    % Ensure helpers are reachable (usually unnecessary, but harmless)
    addpath(pwd);

    % -----------------------------
    % Load segmented signals
    % -----------------------------
    load(dataMatFile, 'data_segmented');  % expects {1}=signal, {2}=table with .Limits and .Label
    if ~iscell(data_segmented) || numel(data_segmented) < 2
        error('Variable "data_segmented" is not in expected format.');
    end
    if ~istable(data_segmented{2}) || ~all(ismember({'Limits','Label'}, data_segmented{2}.Properties.VariableNames))
        error('data_segmented{2} must be a table with Variables: Limits, Label.');
    end
    data = data_segmented{1};

    % Quick segment duration report (helpful for sanity checks)
    duration = data_segmented{2}.Limits(:,2) - data_segmented{2}.Limits(:,1);
    [min_duration, index_min] = min(duration);
    [max_duration, index_max] = max(duration);
    fprintf('Shortest segment: %s (samples %d)\n', char(data_segmented{2}.Label(index_min)), min_duration);
    fprintf('Longest  segment: %s (samples %d)\n', char(data_segmented{2}.Label(index_max)), max_duration);

    % -----------------------------
    % Optional: fix mislabeled rows
    % -----------------------------
    if correct_misslabeled_data
        tbl = data_segmented{2};

        % Exact ranges from the original code
        correct_range_start   = 4562197;
        correct_range_end     = 4572255;
        incorrect_range_start = 4580365;

        row_to_update = find(tbl.Limits(:,1) == correct_range_start & tbl.Limits(:,2) == correct_range_end);
        row_to_remove = find(tbl.Limits(:,1) == incorrect_range_start);

        if ~isempty(row_to_update)
            tbl.Label(row_to_update) = {'SEG_MOTOR_START'};
        end
        if ~isempty(row_to_remove)
            tbl(row_to_remove, :) = [];
        end
        data_segmented{2} = tbl;

        % Build one-hot (binary) vectors per label (as in the original loop)
        M = signalMask(data_segmented{2}); %#ok<NASGU> % (kept for potential plots)
        N = length(data_segmented{1});
        roi_table = data_segmented{2};
        labels = unique(roi_table.Label);

        for ii = 1:numel(labels)
            binSig = zeros(N, 1);
            segs = roi_table.Limits(roi_table.Label == labels(ii), :);
            for jj = 1:size(segs, 1)
                s = max(1, segs(jj,1));
                e = min(N, segs(jj,2));
                binSig(s:e) = 1;
            end
            var_name = matlab.lang.makeValidName(char(labels(ii)));
            eval([var_name ' = binSig;']); % kept for backward-compatibility with rest of the code
        end
        data = data_segmented{1};
    end

    % -----------------------------
    % Iterative "shortening" (non-overlapping shift)
    % -----------------------------
    if shorten_data_to_one_third
        [data_segmented, index_start, index_end, M] = shortenData( ...
            data_segmented, iteration_step, hopSize, total_number_of_iterations);

        if iteration_step == 1
            figure; plotsigroi(M, data_segmented{1}(1:end));
            title('Signal with Regions of Interest (Iteration 1)');
        end
    else
        index_start = 1;
        index_end   = numel(data_segmented{1});
    end

    % -----------------------------
    % Build/adjust SEG_* arrays + SEG_REST
    % -----------------------------
    SEG_variables = {'SEG_MOTOR_START', 'SEG_MOTOR', 'SEG_GENEVA_DRIVE', ...
                     'SEG_DIVERTER', 'SEG_MOTOR_STOPS', 'SEG_TRANSITION_INIT'};

    % SEG_REST starts as all ones and is zeroed wherever any SEG_* is 1
    SEG_REST = ones(1, size(data, 1));

    for v = 1:numel(SEG_variables)
        var_name = SEG_variables{v};

        % Transpose to row and force same length as "data"
        eval([var_name, ' = ', var_name, ''';']);
        eval([var_name, ' = ', var_name, '(:, 1:size(data, 1));']);

        % Zero SEG_REST wherever this segment is active
        eval(['SEG_REST(', var_name, '== 1) = 0;']);
    end

    % -----------------------------
    % Recorded signal (choose vibro/audio as needed)
    % -----------------------------
    RECORDED_SIGNAL = detrend(data) / max(abs(data));  % normalize amplitude
    RECORDED_SIGNAL = RECORDED_SIGNAL';

    % Apply shortening window to both labels and signal
    for v = 1:numel(SEG_variables)
        var_name = SEG_variables{v};
        eval([var_name, ' = ', var_name, ''';']);  % original code style
        if shorten_data_to_one_third
            eval([var_name, ' = ', var_name, '(index_start:index_end);']);
        end
    end
    if shorten_data_to_one_third
        RECORDED_SIGNAL = RECORDED_SIGNAL(index_start:index_end);
    end

    fprintf('Data size after trimming: [%d %d]\n', size(data,1), size(data,2));

    % -----------------------------
    % STFT features (melSpectrogram)
    % -----------------------------
    padded_length = fftLength * ceil(length(RECORDED_SIGNAL) / fftLength);
    recorded_signal_padded = [RECORDED_SIGNAL, zeros(1, padded_length - length(RECORDED_SIGNAL))];

    [RECORDED_SIGNAL_STFT, ~, ~] = melSpectrogram( ...
        recorded_signal_padded', fs, ...
        'Window', win, ...
        'OverlapLength', overlapLength, ...
        'FFTLength', fftLength, ...
        'NumBands', 128);

    stft_length = size(RECORDED_SIGNAL_STFT, 2);
    if stft_length == 0
        error('STFT length is zero. Check input signal or STFT settings.');
    end

    % -----------------------------
    % Map each SEG_* label to STFT domain
    % -----------------------------
    for v = 1:numel(SEG_variables)
        var_name = SEG_variables{v};
        seg_stft = map_to_stft_domain(eval(var_name), hopSize, stft_length);

        if v == 1
            event_count_matrix = zeros(numel(SEG_variables), stft_length);
        end
        event_count_matrix(v, :) = seg_stft;
    end

    % Combine labels per STFT frame (argmax), zero where no events
    [~, combined_seg_stft] = max(event_count_matrix, [], 1);
    combined_seg_stft(sum(event_count_matrix, 1) == 0) = 0;

    % -----------------------------
    % Feature post-processing: extend, log, normalize
    % -----------------------------
    abs_stft = abs(RECORDED_SIGNAL_STFT);

    % context stacking (±7 * numberOfWindows offsets)
    extension_steps = (-7:7) * numberOfWindows;
    abs_stft_extended = zeros(size(abs_stft,1) * numel(extension_steps), size(abs_stft,2));

    for i = 1:size(abs_stft, 2)
        for step_idx = 1:numel(extension_steps)
            idx = i + extension_steps(step_idx);
            if idx >= 1 && idx <= size(abs_stft, 2)
                r0 = (step_idx-1)*size(abs_stft,1) + 1;
                r1 = step_idx*size(abs_stft,1);
                abs_stft_extended(r0:r1, i) = abs_stft(:, idx);
            end
        end
    end

    % log + z-score normalization
    log_stft = log(abs_stft_extended + eps);
    mean_log_stft = mean(log_stft(:));
    std_log_stft  = std(log_stft(:));
    normalized_stft = (log_stft - mean_log_stft) / std_log_stft;

    % -----------------------------
    % Prepare train/val tensors + categorical labels
    % -----------------------------
    [numNodes, trainX_reshaped, valX_reshaped, ~, ~, ~, ~, categorical_trainY, categorical_valY, num_categories] = ...
        prepareDeepLearningData(normalized_stft, combined_seg_stft, splitRatio);

    % -----------------------------
    % Optional undersampling (strategy 1 + 2)
    % -----------------------------
    if UNDERSAMPLE_TRAINING_DATA == undersampling_strategy_values(2) || ...
       UNDERSAMPLE_TRAINING_DATA == undersampling_strategy_values(3)

        % Build categorical classes 0..6
        numCategories = 6;
        cats = arrayfun(@(k) categorical(k), 0:numCategories, 'UniformOutput', false);
        [cat0, cat1, cat2, cat3, cat4, cat5, cat6] = deal(cats{:});

        [undersampledX, undersampledY] = applyFirstUndersamplingStrategy( ...
            trainX_reshaped, categorical_trainY, cat0, cat1, cat2, cat3, cat4, cat5, cat6);

        [finalUndersampledXaux, finalUndersampledYaux] = applySecondUndersamplingStrategy( ...
            undersampledX, undersampledY, UNDERSAMPLE_TRAINING_DATA, undersampling_strategy_values);
    else
        finalUndersampledXaux = trainX_reshaped;
        finalUndersampledYaux = categorical_trainY;
        fprintf('Undersampling not performed. Using original dataset.\n');
    end

    % One-iteration visualization (raw vs downsampled labels)
    if iteration_step == 1
        figure;
        subplot(2,1,1);
        plot(categorical_trainY); title('Original Training Data – one iteration');
        ylabel('Category'); xlabel('Sample Index');

        subplot(2,1,2);
        plot(finalUndersampledYaux); title('Downsampled Training Data – one iteration');
        ylabel('Category'); xlabel('Sample Index');
    end

    % Accumulate across iterations
    finalUndersampledX = cat(4, finalUndersampledX, finalUndersampledXaux);
    finalUndersampledY = cat(1, finalUndersampledY, finalUndersampledYaux);
end

%% ------------------------------------------------------------------------
%  Global stats after iteration loop
% -------------------------------------------------------------------------
fprintf('\n=== Aggregated dataset ===\n');
fprintf('Total samples in finalUndersampledX: %d\n', size(finalUndersampledX, 4));
fprintf('Total labels  in finalUndersampledY: %d\n', numel(finalUndersampledY));

% If you want to shuffle afterwards, uncomment:
% numElements = size(finalUndersampledX, 4);
% idx = randperm(numElements);
% finalUndersampledX = finalUndersampledX(:, :, :, idx);
% finalUndersampledY = finalUndersampledY(idx, :);

figure;
plot(finalUndersampledY); title('Final Undersampled Training Data – all iterations');
ylabel('Category'); xlabel('Sample Index');

%% ------------------------------------------------------------------------
%  Define model, training options, train or load
% -------------------------------------------------------------------------
miniBatchSize = 128;
validationFrequency = ceil(size(finalUndersampledX, 4) / miniBatchSize);
maxEpochs = 6;

layers = [
    imageInputLayer([1, numNodes, 1], 'Normalization', 'none')

    convolution2dLayer([1, 3], 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([1, 2], 'Stride', [1, 2])

    convolution2dLayer([1, 3], 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([1, 2], 'Stride', [1, 2])

    convolution2dLayer([1, 3], 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([1, 2], 'Stride', [1, 2])

    fullyConnectedLayer(128)
    dropoutLayer(0.5)
    fullyConnectedLayer(num_categories)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 3, ...
    'L2Regularization', 0.003, ...
    'MaxEpochs', maxEpochs, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {valX_reshaped, categorical_valY}, ...
    'ValidationFrequency', validationFrequency, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

if train_or_verify_model == option_values(1)
    % -------------------------
    % Train
    % -------------------------
    [net, info] = trainNetwork(finalUndersampledX, finalUndersampledY, layers, options);

    % Save with timestamp
    currentTime = datestr(now, 'dd_mm_yyyy_HH_MM_SS');
    fileName = ['trained_net_matlab_', currentTime, '.mat'];
    save(fileName, 'layers', 'options', 'net', ...
                   'mean_log_stft', 'std_log_stft', 'extension_steps', ...
                   'windowLength', 'fftLength', 'overlapLength', 'win');

    % Optionally keep these in workspace if you need them:
    trainingAccuracy = info.TrainingAccuracy; % #ok<NASGU>
    trainingLoss     = info.TrainingLoss;     % #ok<NASGU>
    validationAccuracy = info.ValidationAccuracy; % #ok<NASGU>
    validationLoss     = info.ValidationLoss;     % #ok<NASGU>

else
    % -------------------------
    % Load pre-trained model
    % -------------------------
    [fileName, path] = uigetfile('*.mat', 'Select a .mat file to load');
    if isequal(fileName, 0)
        error('Loading canceled by user.');
    else
        S = load(fullfile(path, fileName));
        if ~isfield(S, 'net')
            error('Selected file does not contain variable "net".');
        end
        net = S.net;
        fprintf('Loaded file: %s\n', fullfile(path, fileName));
    end
end

%% ------------------------------------------------------------------------
%  Evaluation on validation set
% -------------------------------------------------------------------------
valY_estimate = classify(net, valX_reshaped);

numeric_valY          = double(categorical_valY);
numeric_valY_estimate = double(valY_estimate);

fprintf('\n=== RESULTS (per-class accuracy) ===\n');
[confusionMatrix, ~, ~] = crosstab(numeric_valY, numeric_valY_estimate);

totalPerCategory   = sum(confusionMatrix, 2);
correctPredictions = diag(confusionMatrix);
percentageErrorPerCategory = ((totalPerCategory - correctPredictions) ./ max(totalPerCategory,1)) * 100;

% Print accuracy per category (0-based indexing per your pipeline)
uv = unique(valY_estimate);
for i = 1:numel(uv)
    catId = double(uv(i));    % categorical → numeric value
    % Find row index that matches this category in confusion matrix:
    % (Assumes categories are dense/consistent with numeric coding.)
    rowIdx = find((1:numel(totalPerCategory))' == (catId+1), 1, 'first');
    if isempty(rowIdx), continue; end
    acc = 100 - percentageErrorPerCategory(rowIdx);
    fprintf('Accuracy for category %d: %.2f%%\n', catId, acc);
end

% Plots: Actual vs Predicted and Errors
figure;
set(groot, 'DefaultAxesFontSize', 14);

subplot(2,1,1); hold on;
plot(numeric_valY, 'LineWidth', 1);
plot(numeric_valY_estimate, '--', 'LineWidth', 1);
title('Actual vs Predicted categories');
xlabel('Sample Index'); ylabel('Category');
legend({'Actual', 'Predicted'}, 'Location', 'best'); hold off;

subplot(2,1,2);
plot(numeric_valY - numeric_valY_estimate, 'k', 'LineWidth', 1);
title('Prediction errors'); xlabel('Sample Index'); ylabel('Error');
legend({'Prediction Errors'}, 'Location', 'best');

set(groot, 'DefaultAxesFontSize', 'remove');
