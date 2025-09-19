function num = getUserInput(prompt, dlgtitle, definput, validateFunc)
%GETUSERINPUT Display an input dialog, validate user entry, and return numeric value.
%
%   num = GETUSERINPUT(prompt, dlgtitle, definput, validateFunc)
%
%   INPUTS:
%       prompt       - String shown as the question/prompt inside the dialog box.
%       dlgtitle     - Title of the dialog window (also used in the confirmation printout).
%       definput     - Default input value (string) pre-filled in the dialog.
%       validateFunc - Function handle used to validate input (returns true if valid).
%
%   OUTPUT:
%       num          - Numeric value entered by the user and validated.
%
%   BEHAVIOR:
%   - The dialog has a fixed size of [1 35] characters.
%   - If the user cancels the dialog, a message is printed and the loop terminates.
%   - The input must be convertible to a numeric value (double).
%   - If validation fails (NaN or validateFunc returns false), an error dialog appears,
%     and the user is prompted again until valid input is provided.
%
%   EXAMPLE:
%       validatePositive = @(x) (x > 0);
%       nIter = getUserInput('Enter number of iterations:', 'Iterations', '10', validatePositive);

    % Set dialog dimensions
    dims = [1 35];
    userInputIsValid = false;

    while ~userInputIsValid
        % Open input dialog with default value
        answer = inputdlg(prompt, dlgtitle, dims, {definput});

        % Handle cancel action
        if isempty(answer)
            disp('User cancelled the input.');
            break;
        end

        % Convert string input to numeric
        inputNum = str2double(answer{1});

        % Check if numeric and valid according to validateFunc
        if ~isnan(inputNum) && validateFunc(inputNum)
            userInputIsValid = true;
            num = inputNum;
        else
            % Show modal error dialog if invalid
            uiwait(errordlg('Invalid input. Please try again.', 'Invalid Input', 'modal'));
        end
    end

    % Print confirmation message
    fprintf('%s set to: %.2f\n', dlgtitle, num);
end
