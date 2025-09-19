function [numNodes, trainX_reshaped, valX_reshaped, numericalY, numericalValY, ...
          categoryCounts, categoryCountsVal, categorical_trainY, categorical_valY, num_categories] ...
          = prepareDeepLearningData(normalized_stft, combined_seg_stft, splitRatio)
%PREPAREDEEPLEARNINGDATA Prepare STFT data and labels for deep learning training/validation.
%
%   [numNodes, trainX_reshaped, valX_reshaped, numericalY, numericalValY, ...
%    categoryCounts, categoryCountsVal, categorical_trainY, categorical_valY, num_categories] ...
%    = PREPAREDEEPLEARNINGDATA(normalized_stft, combined_seg_stft, splitRatio)
%
%   INPUTS:
%       normalized_stft   - 2D matrix (numNodes × numSamples), normalized STFT features.
%       combined_seg_stft - Vector of segmentation labels corresponding to STFT frames.
%       splitRatio        - Ratio (0–1) for splitting data into training/validation sets.
%                           Example: 0.5 → 50% training, 50% validation.
%
%   OUTPUTS:
%       numNodes           - Number of STFT nodes (frequency bins).
%       trainX_reshaped    - Training feature data reshaped to 4D format [1, numNodes, 1, numSamples].
%       valX_reshaped      - Validation feature data reshaped to 4D format.
%       numericalY         - Numeric training labels (class indices).
%       numericalValY      - Numeric validation labels (class indices).
%       categoryCounts     - Count of samples per category in training set.
%       categoryCountsVal  - Count of samples per category in validation set.
%       categorical_trainY - Training labels as categorical array.
%       categorical_valY   - Validation labels as categorical array.
%       num_categories     - Total number of unique categories (classes).
%
%   DESCRIPTION:
%   - Splits normalized STFT data into training and validation subsets based
%     on splitRatio.
%   - Converts segmentation labels into categorical and numeric forms.
%   - Reshapes input features to 4D format required by MATLAB Deep Learning Toolbox.
%   - Displays category counts and total number of training samples
%     (before applying any undersampling strategies).
%
%   EXAMPLE:
%       [numNodes, trainX, valX, Ytrain, Yval, counts, countsVal, catY, catYval, nCats] = ...
%           prepareDeepLearningData(stftData, segLabels, 0.7);

    % === Basic parameters ===
    numNodes = size(normalized_stft, 1);          % Number of STFT frequency bins
    numSamples = size(normalized_stft, 2);        % Total number of samples
    splitIdx = round(splitRatio * numSamples);    % Split index for train/validation sets

    % === Split data into training and validation sets ===
    trainX = normalized_stft(:, 1:splitIdx);
    trainY = combined_seg_stft(1:splitIdx);

    valX = normalized_stft(:, splitIdx + 1:end);
    valY = combined_seg_stft(splitIdx + 1:end);

    % === Convert labels to categorical ===
    categorical_trainY = categorical(trainY');
    categorical_valY   = categorical(valY');

    % === Reshape input features to 4D (for Deep Learning Toolbox) ===
    trainX_reshaped = reshape(trainX, [1, numNodes, 1, size(trainX, 2)]);
    valX_reshaped   = reshape(valX, [1, numNodes, 1, size(valX, 2)]);

    % === Determine number of categories ===
    num_categories = numel(unique([trainY'; valY']));

    % === Convert categorical labels to numeric indices ===
    numericalY     = grp2idx(categorical_trainY);
    numericalValY  = grp2idx(categorical_valY);

    % === Count categories in training and validation sets ===
    categoryCounts    = countcats(categorical_trainY);
    categoryCountsVal = countcats(categorical_valY);

    % === Print diagnostic information ===
    for i = 1:numel(categories(categorical_trainY))
        fprintf('Category %d has %d instances in the training set.\n', i, categoryCounts(i));
    end

    fprintf('Total number of samples before undersampling: %d\n', size(trainX_reshaped, 4));
end
