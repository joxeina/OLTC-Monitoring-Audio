function [windowLength, fftLength, hopSize, overlapLength, numberOfWindows, win] = getStftParameters()
%GETSTFTPARAMETERS Define and return parameters for Short-Time Fourier Transform (STFT).
%
%   OUTPUTS:
%       windowLength     - Length of the analysis window (samples).
%       fftLength        - Length of the FFT (samples), determines frequency resolution.
%       hopSize          - Step size (samples) between successive windows.
%       overlapLength    - Number of overlapping samples between adjacent windows.
%       numberOfWindows  - Number of non-overlapping windows that fit in windowLength.
%       win              - Hamming window function of size windowLength.
%
%   NOTE:
%   - With hopSize = windowLength, windows do not overlap.
%   - fftLength is set larger than windowLength to achieve zero-padding 
%     and improve frequency resolution.
%   - Hamming window is used to reduce spectral leakage.

    % === STFT core parameters ===
    windowLength = 1024;                     % Analysis window size in samples
    fftLength    = 2048;                     % FFT length (zero-padding applied)
    hopSize      = 1024;                     % Hop size = shift between windows
    overlapLength = windowLength - hopSize;  % Overlap between adjacent windows

    % Number of non-overlapping windows that fit into the window length
    numberOfWindows = fix(windowLength ./ hopSize);

    % === Window function ===
    % Use periodic Hamming window (suited for spectral analysis)
    win = hamming(windowLength, "periodic");
end
