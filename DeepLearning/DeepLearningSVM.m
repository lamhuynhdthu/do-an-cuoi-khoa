function DeepLearningSVM()
    % Nap du lieu train
     fprintf('\nNap du lieu train ....');
    strFolderDataTrain = fullfile('DataTrain');
    categories = {'0','1','2','3','4','5','6','7','8','9'};
    imdsDataTrain = imageDatastore(fullfile(strFolderDataTrain,categories),'LabelSource','foldernames');
    imdsDataTrain.ReadFcn = @(filename)readAndPreprocessImage(filename);
    net = alexnet();
    featureLayer = 'fc7';
    featuresDataTrain = activations(net,imdsDataTrain,featureLayer,'MiniBatchSize',32,'OutputAs','columns');
    lblDataTrain = imdsDataTrain.Labels;
    
    fprintf('\nXay dung model....');
    classifier = fitcecoc(featuresDataTrain,lblDataTrain,'Learners','Linear','Coding','onevsall','ObservationIn','columns');
    
    fprintf('\nNap du lieu test ....');
    strFolderDataTest = fullfile('DataTest');
    categories = {'0','1','2','3','4','5','6','7','8','9'};
    imdsDataTest = imageDatastore(fullfile(strFolderDataTest,categories),'LabelSource','foldernames');
    imdsDataTest.ReadFcn = @(filename)readAndPreprocessImage(filename);
    featuresDataTest = activations(net,imdsDataTest,featureLayer,'MiniBatchSize',32);
    lblActualDataTest = imdsDataTest.Labels;
    
    % Predict
    fprintf('\nPredict...');
    lblResult = predict(classifier,featuresDataTest);
    nResult = (lblActualDataTest == lblResult);
    nCount = sum(nResult);
    fprintf('\nSo luong mau nhan dang dung: %d\n', nCount);    
end

function Iout = readAndPreprocessImage(filename)
    I = imread(filename);
    if ismatrix(I)
        I = cat(3,I,I,I);
    end
    % Resize the image as required for the CNN
    Iout = imresize(I, [227 227]);
end