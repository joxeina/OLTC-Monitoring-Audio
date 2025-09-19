function [finalUndersampledX, finalUndersampledY, finalUndersampledValX, finalUndersampledValY] = initializeVariables()
%INITIALIZEVARIABLES Initialize empty arrays for training and validation datasets.
%
%   [finalUndersampledX, finalUndersampledY, finalUndersampledValX, finalUndersampledValY] = INITIALIZEVARIABLES()
%
%   OUTPUTS:
%       finalUndersampledX     - Placeholder for undersampled training input data (features).
%       finalUndersampledY     - Placeholder for undersampled training output data (labels).
%       finalUndersampledValX  - Placeholder for undersampled validation input data (features).
%       finalUndersampledValY  - Placeholder for undersampled validation output data (labels).
%
%   DESCRIPTION:
%   This function initializes empty arrays to store both training and
%   validation data after applying undersampling strategies.
%   It provides a clean starting point for accumulating datasets during
%   preprocessing and model preparation steps.
%
%   EXAMPLE:
%       [X, Y, valX, valY] = initializeVariables();
%       % Later in the code, these variables can be populated with data.

    % === Initialization of variables ===
    finalUndersampledX    = [];   % Training features
    finalUndersampledY    = [];   % Training labels
    finalUndersampledValX = [];   % Validation features
    finalUndersampledValY = [];   % Validation labels
end
