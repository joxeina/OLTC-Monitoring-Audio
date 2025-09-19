function [undersampledX, undersampledY] = applyFirstUndersamplingStrategy( ...
    trainX_reshaped, categorical_trainY, cat0, cat1, cat2, cat3, cat4, cat5, cat6)
%APPLYFIRSTUNDERSAMPLINGSTRATEGY Apply undersampling to balance dataset categories.
%
%   [undersampledX, undersampledY] = APPLYFIRSTUNDERSAMPLINGSTRATEGY( ...
%       trainX_reshaped, categorical_trainY, cat0, cat1, cat2, cat3, cat4, cat5, cat6)
%
%   INPUTS:
%       trainX_reshaped   - 4D array of training input data [1, numNodes, 1, numSamples].
%       categorical_trainY- Categorical array of training labels.
%       cat0–cat6         - Category identifiers used to detect transitions.
%
%   OUTPUTS:
%       undersampledX - Training features after undersampling.
%       undersampledY - Training labels after undersampling.
%
%   DESCRIPTION:
%   - Strategy 1 focuses on reducing over-representation of category 0 (idle class).
%   - Two main rules are applied:
%       (1) At transitions between cat0 and other categories (cat1, cat2, cat3, cat5, cat6),
%           only ~10% of intermediate cat0 samples are retained.
%       (2) Long consecutive runs of cat0 (>300 samples) are further undersampled,
%           keeping only ~5% of samples (removing ~95%).
%   - The procedure balances class distribution while keeping transition information.
%
%   EXAMPLE:
%       [Xundersampled, Yundersampled] = applyFirstUndersamplingStrategy(Xtrain, Ytrain, 0,1,2,3,4,5,6);

    fprintf('UNDERSAMPLING STRATEGY 1 STARTS\n');

    % === Initialize keepIndices (logical mask, true = keep sample) ===
    keepIndices = true(length(categorical_trainY), 1);

    % === Rule (1): Handle category transitions involving cat0 ===
    for i = 2:length(categorical_trainY)
        if ((categorical_trainY(i-1) == cat1 && categorical_trainY(i) == cat0) || (categorical_trainY(i-1) == cat0 && categorical_trainY(i) == cat2)) || ...
           ((categorical_trainY(i-1) == cat2 && categorical_trainY(i) == cat0) || (categorical_trainY(i-1) == cat0 && categorical_trainY(i) == cat3)) || ...
           ((categorical_trainY(i-1) == cat5 && categorical_trainY(i) == cat0) || (categorical_trainY(i-1) == cat0 && categorical_trainY(i) == cat6))

            % Mark start of a cat0 sequence
            startIdx = i;
            while i <= length(categorical_trainY) && categorical_trainY(i) == cat0
                i = i + 1;
            end
            endIdx = i - 1;

            % Exclude first and last 3 samples from downsampling
            adjustedStartIdx = max(startIdx + 3, startIdx);
            adjustedEndIdx   = min(endIdx - 3, endIdx);

            % Randomly keep ~10% of cat0 samples in this range
            range = adjustedStartIdx:adjustedEndIdx;
            numToRetain = ceil(length(range) / 10);
            indicesToKeep = randsample(range, numToRetain);

            % Drop all cat0 in range except selected ones
            keepIndices(range) = false;
            keepIndices(indicesToKeep) = true;
        end
    end

    % === Rule (2): Downsample long consecutive runs of cat0 (>300 samples) ===
    category0StartIdx = find(categorical_trainY == cat0, 1, 'first');
    while ~isempty(category0StartIdx) && category0StartIdx <= length(categorical_trainY)
        category0EndIdx = category0StartIdx;
        while category0EndIdx <= length(categorical_trainY) && categorical_trainY(category0EndIdx) == cat0
            category0EndIdx = category0EndIdx + 1;
        end
        category0EndIdx = category0EndIdx - 1;

        % If run length > 300, keep only ~5% of samples
        if (category0EndIdx - category0StartIdx + 1) > 300
            downsampleRange = category0StartIdx:category0EndIdx;
            indicesToRemove = randsample(downsampleRange, floor(length(downsampleRange) * 19 / 20));
            keepIndices(indicesToRemove) = false;
        end

        % Move to next occurrence of cat0
        nextIdx = find(categorical_trainY(category0EndIdx+1:end) == cat0, 1, 'first');
        if isempty(nextIdx)
            break;
        end
        category0StartIdx = nextIdx + category0EndIdx;
    end

    % === Apply undersampling to training data ===
    undersampledX = trainX_reshaped(:,:,:,keepIndices);
    undersampledY = categorical_trainY(keepIndices);

    % === Display category statistics after undersampling ===
    uniqueCategories = unique(undersampledY);
    categoryCountsAfterFirst = histcounts(undersampledY, uniqueCategories);

    for i = 1:length(uniqueCategories)
        fprintf('Category %s has %d instances after the first undersampling strategy.\n', ...
            char(uniqueCategories(i)), categoryCountsAfterFirst(i));
    end

    % Print total number of remaining samples
    totalUndersampled = sum(keepIndices);
    fprintf('Total number of samples after first undersampling strategy: %d\n', totalUndersampled);

    fprintf('UNDERSAMPLING STRATEGY 1 COMPLETED\n');
end
