function RawKNN(k,d)

    % Nap du lieu train
    fprintf('\nNap du lieu train...');
    strDataTrain = './train-images.idx3-ubyte';
    strLabelTrain = './train-labels.idx1-ubyte';
    [imgDataTrain, lblDataTrain] = loadData(strDataTrain, strLabelTrain);
    
    % Xay dung mo hinh voi KNN
    fprintf('\nXay dung model...');
    Mdl = fitcknn(imgDataTrain', lblDataTrain, 'NumNeighbors',k,'Distance',d);
    
    % Nap du lieu test
    fprintf('\nNap du lieu test...');
    strDataTest = './t10k-images.idx3-ubyte';
    strLabelTest = './t10k-labels.idx1-ubyte';
    [imgDataTest, lblDataTest] = loadData(strDataTest, strLabelTest);
    
    % Predict
    fprintf('\nPredict...');
    lblResult = predict(Mdl,imgDataTest');
    nResult = (lblResult == lblDataTest);
    nCount = sum(nResult);
    fprintf('\nSo luong mau dung: %d\n', nCount);
    
end