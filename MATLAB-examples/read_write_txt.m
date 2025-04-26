function read_write_txt(folderPath, outputFileName)
    % Function to recursively read all .txt files in a folder (and subfolders)
    % and write their contents to an output file.

    % Check if the folder exists
    if ~isfolder(folderPath)
        error('The specified folder does not exist.');
    end

    % Open the output file for writing
    outputFileID = fopen(outputFileName, 'w');
    if outputFileID == -1
        error('Could not open output file for writing.');
    end

    % Start processing the folder
    try
        processFolder(folderPath, outputFileID);
    catch ME
        fclose(outputFileID);
        rethrow(ME);
    end

    % Close the output file
    fclose(outputFileID);
    fprintf('Contents written to %s successfully.\n', outputFileName);
end

function processFolder(folderPath, outputFileID)
    % Get all .txt files and subfolders in the current folder
    files = dir(fullfile(folderPath, '*.txt'));
    subfolders = dir(folderPath);

    % Write the contents of each .txt file to the output file
    for i = 1:length(files)
        filePath = fullfile(folderPath, files(i).name);
        fprintf('Reading file: %s\n', filePath);
        writeFileContents(filePath, outputFileID);
    end

    % Recursively process subfolders
    for i = 1:length(subfolders)
        if subfolders(i).isdir && ~ismember(subfolders(i).name, {'.', '..'})
            subfolderPath = fullfile(folderPath, subfolders(i).name);
            processFolder(subfolderPath, outputFileID);
        end
    end
end

function writeFileContents(filePath, outputFileID)
    % Open the .txt file for reading
    fileID = fopen(filePath, 'r');
    if fileID == -1
        warning('Could not open file: %s. Skipping.', filePath);
        return;
    end

    % Write the file's contents to the output file
    while ~feof(fileID)
        line = fgets(fileID);
        if ischar(line)
            fprintf(outputFileID, '%s', line);
        end
    end

    % Add a separator or newline after each file
    fprintf(outputFileID, '\n\n--- End of File: %s ---\n\n', filePath);

    % Close the input file
    fclose(fileID);
end