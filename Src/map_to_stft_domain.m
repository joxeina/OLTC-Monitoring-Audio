function seg_stft = map_to_stft_domain(seg_time, hop_size, stft_length)
%MAP_TO_STFT_DOMAIN Map a time-domain segmentation signal to STFT domain.
%
%   seg_stft = MAP_TO_STFT_DOMAIN(seg_time, hop_size, stft_length)
%
%   INPUTS:
%       seg_time    - Binary time-domain segmentation vector (e.g., SEG_ signal).
%       hop_size    - Step size (samples) used in STFT processing.
%       stft_length - Target length of STFT-domain representation.
%
%   OUTPUT:
%       seg_stft    - Binary vector in the STFT domain. Each element indicates
%                     whether the corresponding STFT frame contains an event
%                     (1 = active, 0 = inactive).
%
%   DESCRIPTION:
%   - The function downsamples a time-domain segmentation signal to match the
%     STFT frame indices defined by hop_size.
%   - For each STFT frame interval, the function checks whether any event
%     is present in seg_time. If yes, that STFT bin is marked as 1.
%   - The resulting vector can be used to align segmentation labels with STFT
%     feature representations.
%
%   EXAMPLE:
%       seg_time = [0 0 1 1 0 0 1 0 0 0]; % binary activity
%       hop_size = 2;
%       stft_len = 5;
%       seg_stft = map_to_stft_domain(seg_time, hop_size, stft_len);
%
%       % seg_stft = [0 1 0 1 0]
%

    % Initialize STFT domain vector with zeros
    seg_stft = zeros(1, stft_length);

    % Determine index positions for STFT frames
    seg_indices = 1:hop_size:length(seg_time);

    % Iterate over frame intervals
    for i = 1:length(seg_indices)-1
        % Mark 1 if any non-zero element is found within the frame range
        seg_stft(i) = any(seg_time(seg_indices(i):seg_indices(i+1)-1));
    end
end
