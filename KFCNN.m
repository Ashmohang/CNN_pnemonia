% Pneumonia Detection using CNN with 2-Fold Cross-Validation
% Author: Ashwin Mohan

clc; clear; close all;

% Set your dataset path here (each class in its own folder: NORMAL, PNEUMONIA)
dataPath = 'your/full/path/to/chest_xray/';

% Load dataset
imds = imageDatastore(dataPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Resize input images to a consistent format
inputSize = [227 227];  % You can change this as needed

% Split dataset using 2-fold cross-validation
k = 2;
cv = cvpartition(imds.Labels, 'KFold', k);
accList = zeros(k,1);

for fold = 1:k
    fprintf('\nRunning Fold %d of %d...\n', fold, k);

    % Split training and testing
    imdsTrain = subset(imds, training(cv, fold));
    imdsTest  = subset(imds, test(cv, fold));

    % Augment training images
    augTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
        'DataAugmentation', imageDataAugmenter( ...
            'RandRotation', [-10 10], ...
            'RandXReflection', true, ...
            'RandXTranslation', [-5 5], ...
            'RandYTranslation', [-5 5]));

    % Preprocess test images
    augTest = augmentedImageDatastore(inputSize, imdsTest);

    % Define CNN architecture
    layers = [
        imageInputLayer([inputSize 3],'Name','input')

        convolution2dLayer(3,8,'Padding','same','Name','conv1')
        batchNormalizationLayer('Name','bn1')
        reluLayer('Name','relu1')
        maxPooling2dLayer(2,'Stride',2,'Name','maxpool1')

        convolution2dLayer(3,16,'Padding','same','Name','conv2')
        batchNormalizationLayer('Name','bn2')
        reluLayer('Name','relu2')
        maxPooling2dLayer(2,'Stride',2,'Name','maxpool2')

        convolution2dLayer(3,32,'Padding','same','Name','conv3')
        batchNormalizationLayer('Name','bn3')
        reluLayer('Name','relu3')
        maxPooling2dLayer(2,'Stride',2,'Name','maxpool3')

        fullyConnectedLayer(2,'Name','fc')
        softmaxLayer('Name','softmax')
        classificationLayer('Name','output')];

    % Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 10, ...
        'InitialLearnRate', 1e-4, ...
        'ValidationData', augTest, ...
        'ValidationFrequency', 30, ...
        'Verbose', false, ...
        'Plots','none');

    % Train network
    net = trainNetwork(augTrain, layers, options);

    % Predict on test set
    predictedLabels = classify(net, augTest);
    trueLabels = imdsTest.Labels;

    % Accuracy
    accuracy = mean(predictedLabels == trueLabels);
    accList(fold) = accuracy;

    % Display confusion matrix
    figure;
    cm = confusionchart(trueLabels, predictedLabels);
    cm.Title = sprintf('Confusion Matrix - Fold %d', fold);
    cm.RowSummary = 'row-normalized';
    cm.ColumnSummary = 'column-normalized';

    fprintf('Fold %d Accuracy: %.2f%%\n', fold, accuracy * 100);
end

% Final report
fprintf('\n--- Final Results ---\n');
fprintf('Average Accuracy over %d folds: %.2f%%\n', k, mean(accList) * 100);
% Paste your full MATLAB CNN code here
