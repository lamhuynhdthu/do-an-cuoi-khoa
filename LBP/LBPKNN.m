function LBPKNN()
    % Nap du lieu train
    fprintf('\nNap du lieu train...');
    strDataTrain = 'train-images.idx3-ubyte';
    strLabelTrain = 'train-labels.idx1-ubyte';
    [imgDataTrain, lblDataTrain] = loadData(strDataTrain, strLabelTrain);
    
    % Trich chon dac trung tap du lieu train
    featuresDataTrain = ExtractFeaturesLBP(imgDataTrain);
    
    % Xay dung model voi KNN
    Mdl = fitcknn(featuresDataTrain', lblDataTrain);
    
    % Nap du lieu test
    fprintf('\nNap du lieu test...');
    strDataTest = 't10k-images.idx3-ubyte';
    strLabelTest = 't10k-labels.idx1-ubyte';
    [imgDataTest, lblDataTest] = loadData(strDataTest, strLabelTest);
    
    % Trich chon dac trung tap du lieu test
    featuresDataTest = ExtractFeaturesLBP(imgDataTest);
    
    % Predict
    fprintf('\nPredict...');
    lblResult = predict(Mdl, featuresDataTest');
    nResult = (lblResult == lblDataTest);
    nCount = sum(nResult);
    fprintf('\nSo luong mau dung: %d\n', nCount);    
end