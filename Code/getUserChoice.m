function choice = getUserChoice(options, values, prompt, defaultMsg)
%GETUSERCHOICE Display a list dialog and return the user’s selection.
%
%   choice = GETUSERCHOICE(options, values, prompt, defaultMsg)
%
%   INPUTS:
%       options    - Cell array of option labels to display in the list dialog.
%       values     - Array of corresponding values for each option.
%       prompt     - String shown as the dialog prompt.
%       defaultMsg - Message displayed if the user cancels (default index=1 used).
%
%   OUTPUT:
%       choice     - The value associated with the selected option.
%
%   NOTES:
%   - Only single selection is allowed.
%   - If the user cancels the dialog, the first option is selected by default,
%     and defaultMsg is printed to the command window.

    % Open a dialog window for user selection
    [indx, tf] = listdlg( ...
        'PromptString', prompt, ...   % Message at the top of the dialog
        'SelectionMode', 'single', ...% Only one option can be chosen
        'ListString', options);       % List of available options

    % Handle case when user cancels (tf = 0)
    if ~tf
        disp(defaultMsg);             % Print fallback message
        indx = 1;                     % Default to first option
    end

    % Map the selected index to the corresponding value
    choice = values(indx);

    % Display chosen option in the command window
    fprintf('Selected option: %s\n', options{indx});
end
