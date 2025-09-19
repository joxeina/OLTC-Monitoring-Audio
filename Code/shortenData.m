function [data_segmented, index_start, index_end, M] = shortenData(data_segmented, iteration_step, hopSize, total_number_of_iterations)
%SHORTENDATA Trim segmented data for a specific iteration and update signal mask.
%
%   [data_segmented, index_start, index_end, M] = SHORTENDATA(data_segmented, iteration_step, hopSize, total_number_of_iterations)
%
%   INPUTS:
%       data_segmented            - Cell array containing:
%                                     {1} → signal data (vector),
%                                     {2} → segmentation table with event limits.
%       iteration_step            - Current iteration index (1-based).
%       hopSize                   - Hop size in samples (used in STFT).
%       total_number_of_iterations- Total number of iterations defined by user.
%
%   OUTPUTS:
%       data_segmented - Updated cell array after trimming (signal + segmentation table).
%       index_start    - Start index used for trimming the signal.
%       index_end      - End index used for trimming the signal.
%       M              - Signal mask (generated from updated segmentation table).
%
%   DESCRIPTION:
%   - Calculates start and end indices for trimming the signal
%     based on iteration step and hop size.
%   - Trims signal data and removes segmentation rows outside the new limits.
%   - Generates an updated signal mask from the trimmed segmentation data.
%
%   NOTE:
%   - The constant `8364060` defines the reference maximum index for trimming.
%     This should match the dataset length (can be generalized if needed).
%
%   EXAMPLE:
%       [segData, idxStart, idxEnd, mask] = shortenData(segData, 2, 1024, 4);
%

    % === Calculate start and end indices for trimming ===
    index_start = 1 + (iteration_step - 1) * floor(floor(hopSize / total_number_of_iterations));
    index_end   = 8364060 + (iteration_step - 1 - hopSize) * floor(hopSize / total_number_of_iterations);

    % === Display debug info ===
    disp(['Index start: ', num2str(index_start)]);
    disp(['Index end:   ', num2str(index_end)]);
    disp('....................................');
    disp(['Size of data before trimming: ', num2str(size(data_segmented{1}))]);

    % === Trim the signal ===
    data_segmented{1} = data_segmented{1}(index_start:index_end);

    % === Remove segmentation rows outside new limits ===
    rows_to_remove = data_segmented{2}.Limits(:, 1) > index_end | data_segmented{2}.Limits(:, 2) > index_end;
    data_segmented{2}(rows_to_remove, :) = [];

    % === Generate updated signal mask ===
    M = signalMask(data_segmented{2});
end
