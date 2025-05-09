% Define the directory containing your AMC files
amcDirectory = 'D:\University\FYP\Implementation\HDM_01-01_amc\DATA';
addpath(genpath('D:\University\FYP\Implementation\HDM05-Parser\HDM05-Parser\parser'))
addpath(genpath('D:\University\FYP\Implementation\HDM05-Parser\HDM05-Parse'))


% r\animate'))
addpath('D:\University\FYP\Implementation\HDM05-Parser\HDM05-Parser\quaternions')
% Define the directory containing your AMC files

% Get a list of all AMC files in the directory
amcFiles = dir(fullfile(amcDirectory, '*.amc'));

% Initialize an empty matrix to store the concatenated data
concatenatedData = [];

% Loop through each AMC file and extract motion data
for i = 1:length(amcFiles)
    % Construct the full file path
    amcFilePath = fullfile(amcDirectory, amcFiles(i).name);
    
    % Read the motion data from the AMC file
    D = amc_to_matrix(amcFilePath);
    
    % Append the motion data to the concatenatedData matrix
    concatenatedData = [concatenatedData; D];
end

% Create a table from the concatenated data
motionDataTable = array2table(concatenatedData);

% Save the table to a CSV file
writetable(motionDataTable, 'All_Motions_HDMO5.csv');

disp('Data from all AMC files has been appended and saved as motion_data.csv');
