% =========================================================================
% Elin State Tagging – Multi-Network Validation Script (single-file)
% =========================================================================
% - Lets you pick N trained networks (.mat files with variable `net`)
% - Builds features the same way as the training script
% - Classifies with each network; if >=3 networks, adds "Combined" (majority vote)
% - Reports per-category accuracy per iteration and aggregates stats
%
% Assumes helper functions exist on MATLAB path:
%   getStftParameters, getUserInput, getUserChoice, shortenData,
%   map_to_stft_domain, prepareDeepLearningData
%
% Author: <you>
% ========================================================================

close all; clearvars; clc;
rng('default');            % (optional) reproducible randsample
fs = 34133;                % sampling rate (consistent with training)

%% ------------------------------------------------------------------------
%  STFT parameters & user inputs
% -------------------------------------------------------------------------
[windowLength, fftLength, hopSize, overlapLength, numberOfWindows, win] = getStftParameters();

% How many networks will you test?
numNetworks = getUserInput( ...
    'Enter the number of networks (must be > 0, < 10, and an integer):', ...
    'Input for numNetworks', '3', @(x) x > 0 && x < 10 && mod(x, 1) == 0);

% Pick each network file; store into net1, net2, ...
for netIndex = 1:numNetworks
    [fileName, path] = uigetfile('*.mat', sprintf('Select the .mat file for network %d', netIndex));
    if isequal(fileName, 0)
        disp('Loading canceled.');
        numNetworks = netIndex - 1;  % adjust to how many we actually loaded
        break;
    else
        S = load(fullfile(path, fileName));
        assert(isfield(S, 'net'), 'Selected file does not contain variable "net".');
        eval(sprintf('net%d = S.net;', netIndex)); % keep the dynamic variables style
        fprintf('Loaded Network %d: %s\n', netIndex, fullfile(path, fileName));
    end
end
if numNetworks == 0
    error('No networks loaded. Aborting.');
end

% Iterations & split choices (same rules as training)
total_number_of_iterations = getUserInput( ...
    'Enter a total number of iterations (must be divisible by 2 and not exceed windowLength):', ...
    'Input for Iterations', '4', @(x) mod(x, 2) == 0 && x <= windowLength);

splitRatio = getUserInput( ...
    'Enter the training/validation split ratio (between 0.1 and 0.9):', ...
    'Input for Training/Validation Split Ratio', '0.5', @(x) x >= 0.1 && x <= 0.9);

% Optional mislabeled-data correction
correct_misslabeled_data = getUserChoice( ...
    {'No', 'Yes'}, [0, 1], 'Correct mislabeled data?', ...
    'No option selected, defaulting to No.');

shorten_data_to_one_third = 1;

%% ------------------------------------------------------------------------
%  Prepare containers for per-iteration stats
% -------------------------------------------------------------------------
numCategories = 7;  % pipeline uses categories 0..6  (7 categories)
addCombined = (numNetworks >= 3);           % if >=3 nets, we add "Combined" as an extra model
totalModels = numNetworks + double(addCombined);

% Each cell holds a vector of error% over iterations (one vector per model)
errorsPerCategoryPerIteration = cell(numCategories, totalModels);

%% ------------------------------------------------------------------------
%  Resolve data path (../data/...), load once per iteration
% -------------------------------------------------------------------------
projectRoot = fileparts(fileparts(mfilename('fullpath'))); % src/ -> up -> project root
dataMatFile = fullfile(projectRoot, 'data', 'segmented_data_elin_48kHz_pos1-27.mat');
if ~isfile(dataMatFile)
    error('Data file not found: %s', dataMatFile);
end

%% ------------------------------------------------------------------------
%  Iterations loop
% -------------------------------------------------------------------------
for iteration_step = 1:total_number_of_iterations
    fprintf('\n=== Iteration %d / %d ===\n', iteration_step, total_number_of_iterations);

    % --- Load segmented data (expects data_segmented{1}=signal, {2}=table with Limits & Label)
    load(dataMatFile, 'data_segmented');
    if ~iscell(data_segmented) || numel(data_segmented) < 2
        error('Variable "data_segmented" is not in expected format.');
    end
    if ~istable(data_segmented{2}) || ~all(ismember({'Limits','Label'}, data_segmented{2}.Properties.VariableNames))
        error('data_segmented{2} must be a table with Variables: Limits, Label.');
    end
    data = data_segmented{1};

    % Quick segment duration report
    duration = data_segmented{2}.Limits(:,2) - data_segmented{2}.Limits(:,1);
    [min_duration, index_min] = min(duration);
    [max_duration, index_max] = max(duration);
    fprintf('Shortest segment: %s (samples %d)\n', char(data_segmented{2}.Label(index_min)), min_duration);
    fprintf('Longest  segment: %s (samples %d)\n', char(data_segmented{2}.Label(index_max)), max_duration);

    % --- Optional mislabeled correction (as in the training script)
    if correct_misslabeled_data
        tbl = data_segmented{2};

        correct_range_start   = 4562197;
        correct_range_end     = 4572255;
        incorrect_range_start = 4580365;

        row_to_update = find(tbl.Limits(:,1) == correct_range_start & tbl.Limits(:,2) == correct_range_end);
        row_to_remove = find(tbl.Limits(:,1) == incorrect_range_start);

        if ~isempty(row_to_update), tbl.Label(row_to_update) = {'SEG_MOTOR_START'}; end
        if ~isempty(row_to_remove), tbl(row_to_remove, :) = []; end

        data_segmented{2} = tbl;

        % Build binary vectors for each label (kept for compatibility)
        M = signalMask(data_segmented{2}); %#ok<NASGU>
        N = length(data_segmented{1});
        roi_table = data_segmented{2};
        labels = unique(roi_table.Label);
        for ii = 1:numel(labels)
            binary_signal = zeros(N, 1);
            segs = roi_table.Limits(roi_table.Label == labels(ii), :);
            for jj = 1:size(segs, 1)
                s = max(1, segs(jj,1)); e = min(N, segs(jj,2));
                binary_signal(s:e) = 1;
            end
            var_name = matlab.lang.makeValidName(char(labels(ii)));
            eval([var_name ' = binary_signal;']);
        end
        data = data_segmented{1};
    end

    % --- Shorten to the iteration slice (non-overlapping hop)
    if shorten_data_to_one_third
        [data_segmented, index_start, index_end, M] = ...
            shortenData(data_segmented, iteration_step, hopSize, total_number_of_iterations);
        if iteration_step == 1
            figure; plotsigroi(M, data_segmented{1}(1:end));
            title('Signal with Regions of Interest (Iteration 1)');
        end
    else
        index_start = 1;
        index_end   = numel(data_segmented{1});
    end

    % --- Build SEG_* + SEG_REST (keep the eval-based pipeline)
    SEG_variables = {'SEG_MOTOR_START','SEG_MOTOR','SEG_GENEVA_DRIVE', ...
                     'SEG_DIVERTER','SEG_MOTOR_STOPS','SEG_TRANSITION_INIT'};

    SEG_REST = ones(1, size(data, 1));
    for v = 1:numel(SEG_variables)
        var_name = SEG_variables{v};
        eval([var_name, '=', var_name, ''';']);
        eval([var_name, '=', var_name, '(:, 1:size(data, 1));']);
        eval(['SEG_REST(', var_name, '== 1) = 0;']);
    end

    % --- Recorded signal choice (vibro/audio); normalize amplitude
    RECORDED_SIGNAL = detrend(data) / max(abs(data));
    RECORDED_SIGNAL = RECORDED_SIGNAL';

    % Apply shortening to labels + signal
    for v = 1:numel(SEG_variables)
        var_name = SEG_variables{v};
        eval([var_name, ' = ', var_name, ''';']);
        if shorten_data_to_one_third
            eval([var_name, ' = ', var_name, '(index_start:index_end);']);
        end
    end
    if shorten_data_to_one_third
        RECORDED_SIGNAL = RECORDED_SIGNAL(index_start:index_end);
    end

    fprintf('Data size after trimming: [%d %d]\n', size(data,1), size(data,2));

    % --- Mel spectrogram features
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

    % --- Map SEG_* masks to STFT frames
    for v = 1:numel(SEG_variables)
        var_name = SEG_variables{v};
        seg_stft = map_to_stft_domain(eval(var_name), hopSize, stft_length);
        if v == 1
            event_count_matrix = zeros(numel(SEG_variables), stft_length);
        end
        event_count_matrix(v, :) = seg_stft;
    end

    [~, combined_seg_stft] = max(event_count_matrix, [], 1);
    combined_seg_stft(sum(event_count_matrix, 1) == 0) = 0;

    % --- Extend context, log, normalize
    abs_stft = abs(RECORDED_SIGNAL_STFT);
    extension_steps = (-7:7) * numberOfWindows;
    abs_stft_extended = zeros(size(abs_stft,1)*numel(extension_steps), size(abs_stft,2));
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

    log_stft = log(abs_stft_extended + eps);
    mean_log_stft = mean(log_stft(:)); % #ok<NASGU>  % kept if you want to save later
    std_log_stft  = std(log_stft(:));  % #ok<NASGU>
    normalized_stft = (log_stft - mean_log_stft) / std_log_stft;

    % --- Prepare deep learning data (split + reshape)
    [numNodes, ~, valX_reshaped, ~, ~, ~, ~, ~, categorical_valY, ~] = ...
        prepareDeepLearningData(normalized_stft, combined_seg_stft, splitRatio);

    % --- Classify with each network
    valY_estimates = cell(1, numNetworks);
    for iNet = 1:numNetworks
        valY_estimates{iNet} = classify(eval(['net', num2str(iNet)]), valX_reshaped);
    end

    % If >=3 networks, add majority vote as "Combined"
    if addCombined
        valY_estimate_combined = mode(cat(3, valY_estimates{:}), 3);
        valY_estimates{end+1} = valY_estimate_combined;  % append as last model
    end

    % --- Convert to numeric for analysis
    numeric_valY = double(categorical_valY);
    numeric_valY_estimates = cellfun(@double, valY_estimates, 'UniformOutput', false);

    % --- Per-model analysis (confusion mat, per-class error%)
    networkNames = arrayfun(@(i) sprintf('Network %d', i), 1:numNetworks, 'UniformOutput', false);
    if addCombined
        networkNames{end+1} = 'Combined';
    end

    for m = 1:numel(numeric_valY_estimates)
        fprintf('RESULTS for %s\n', networkNames{m});

        % Confusion matrix & per-category accuracy
        [cm, ~, ~] = crosstab(numeric_valY, numeric_valY_estimates{m});
        totalPerCategory = sum(cm, 2);
        correctPred = diag(cm);
        % Protect against division by zero
        totalPerCategory(totalPerCategory==0) = 1;
        percError = ((totalPerCategory - correctPred) ./ totalPerCategory) * 100;

        for ci = 1:length(unique(numeric_valY_estimates{m}))
            fprintf('Accuracy for category %d in %s: %.2f%%\n', ci-1, networkNames{m}, 100 - percError(ci));
        end

        % Append error% for this iteration -> to aggregate later
        for ci = 1:numCategories
            if iteration_step == 1
                errorsPerCategoryPerIteration{ci, m} = percError(min(ci, numel(percError)));
            else
                errorsPerCategoryPerIteration{ci, m} = ...
                    [errorsPerCategoryPerIteration{ci, m}, percError(min(ci, numel(percError)))];
            end
        end
    end

    % --- First-iteration visual comparison (Actual vs each Predicted)
    if iteration_step == 1
        figure; set(groot, 'DefaultAxesFontSize', 14);
        for m = 1:numel(numeric_valY_estimates)
            subplot(numel(numeric_valY_estimates), 1, m); hold on;
            plot(numeric_valY, 'b', 'LineWidth', 1);
            plot(numeric_valY_estimates{m}, 'r--', 'LineWidth', 1);
            title(['Actual vs Predicted categories for ', networkNames{m}]);
            xlabel('Sample Index'); ylabel('Category');
            legend({'Actual categories', 'Predicted categories'}, 'Location', 'best');
            hold off;
        end
        set(groot, 'DefaultAxesFontSize', 'remove');
    end
end

%% ------------------------------------------------------------------------
%  Aggregate results across iterations (mean & std per category/model)
% -------------------------------------------------------------------------
meanErrors = zeros(numCategories, totalModels);
meanAccuracy = zeros(numCategories, totalModels);
stdDeviationAccuracy = zeros(numCategories, totalModels);
stdDeviationErrors = zeros(numCategories, totalModels);

for m = 1:totalModels
    for ci = 1:numCategories
        errs = errorsPerCategoryPerIteration{ci, m};
        meanErrors(ci, m) = mean(errs);
        accVec = 100 - errs;
        meanAccuracy(ci, m) = mean(accVec);
        stdDeviationAccuracy(ci, m) = std(accVec);
        stdDeviationErrors(ci, m) = std(errs);
    end
end

% Plot mean accuracy + std dev per model
figure;

subplot(2,1,1);
bar(meanAccuracy, 'grouped');
title('Mean Accuracy by Categories and Networks');
xlabel('Category'); ylabel('Mean Accuracy (%)');
netLegend = arrayfun(@(i) sprintf('Network %d', i), 1:numNetworks, 'UniformOutput', false);
if addCombined, netLegend{end+1} = 'Combined'; end
legend(netLegend{:}, 'Location', 'best');
ylim([0 100]);

subplot(2,1,2);
bar(stdDeviationAccuracy, 'grouped');
title('Standard Deviation of Accuracy by Categories and Networks');
xlabel('Category'); ylabel('Std Dev of Accuracy (%)');
legend(netLegend{:}, 'Location', 'best');
ylim([0, max(stdDeviationAccuracy(:))+5]);
