function [finalUndersampledXaux, finalUndersampledYaux] = applySecondUndersamplingStrategy( ...
    undersampledX, undersampledY, UNDERSAMPLE_TRAINING_DATA, undersampling_strategy_values)
%APPLYSECONDUNDERSAMPLINGSTRATEGY Apply secondary undersampling to balance dataset categories.
%
%   [finalUndersampledXaux, finalUndersampledYaux] = APPLYSECONDUNDERSAMPLINGSTRATEGY( ...
%       undersampledX, undersampledY, UNDERSAMPLE_TRAINING_DATA, undersampling_strategy_values)
%
%   INPUTS:
%       undersampledX               - 4D array of features after first undersampling.
%       undersampledY               - Categorical labels after first undersampling.
%       UNDERSAMPLE_TRAINING_DATA   - Selected undersampling strategy (user input).
%       undersampling_strategy_values- Array of possible undersampling strategy IDs.
%
%   OUTPUTS:
%       finalUndersampledXaux - Features after second undersampling (balanced).
%       finalUndersampledYaux - Labels after second undersampling.
%
%   DESCRIPTION:
%   - Strategy 2 ensures that all categories are balanced according to the
%     **second most populated category** (not the maximum).
%   - For each category:
%       * If its size > second most populous size → random downsampling.
%       * Otherwise → keep all samples.
%   - This prevents extreme imbalance when one category dominates heavily.
%
%   BEHAVIOR:
%   - If UNDERSAMPLE_TRAINING_DATA ~= undersampling_strategy_values(2),
%     the function simply returns results of Strategy 1 without changes.
%
%   EXAMPLE:
%       [Xfinal, Yfinal] = applySecondUndersamplingStrategy(X1, Y1, 2, [1 2 3]);

    if UNDERSAMPLE_TRAINING_DATA == undersampling_strategy_values(2)
        fprintf('UNDERSAMPLING STRATEGY 2 STARTS\n');

        % === Count samples per category ===
        categoryCounts = histcounts(undersampledY, unique(undersampledY));

        % Sort counts descending to find the second most populous category
        sortedCounts = sort(categoryCounts, 'descend');
        secondMostPopulousCount = sortedCounts(min(2, length(sortedCounts)));

        % Initialize mask for final undersampling
        finalKeepIndices = false(size(undersampledY));

        % === Apply balancing across categories ===
        uniqueCategories = unique(undersampledY);
        for category = uniqueCategories'
            categoryIndices = find(undersampledY == category);
            selectedIndices = categoryIndices;

            % If category is larger than second-most populous → downsample
            if length(categoryIndices) > secondMostPopulousCount
                selectedIndices = randsample(categoryIndices, secondMostPopulousCount);
            end

            finalKeepIndices(selectedIndices) = true;
        end

        % === Apply final undersampling ===
        finalUndersampledXaux = undersampledX(:,:,:,finalKeepIndices);
        finalUndersampledYaux = undersampledY(finalKeepIndices);

        % === Display stats ===
        totalFinalUndersampled = sum(finalKeepIndices);
        fprintf('Total number of samples after second undersampling strategy: %d\n', totalFinalUndersampled);

        uniqueCategoriesFinal = unique(finalUndersampledYaux);
        finalCategoryCounts = histcounts(finalUndersampledYaux, uniqueCategoriesFinal);

        for i = 1:length(uniqueCategoriesFinal)
            fprintf('Category %s has %d instances after the final undersampling strategy.\n', ...
                char(uniqueCategoriesFinal(i)), finalCategoryCounts(i));
        end

        fprintf('UNDERSAMPLING STRATEGY 2 COMPLETED\n');

    else
        % If strategy 2 is not selected → return inputs unchanged
        finalUndersampledXaux = undersampledX;
        finalUndersampledYaux = undersampledY;
    end
end
