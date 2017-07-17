function [featuresDataTrain] = ExtractFeaturesLBP(imgDataTrain)
    imgI1D = imgDataTrain(:,1);
    imgI2D = reshape(imgI1D, 28, 28);
    featureVector = extractLBPFeatures(imgI2D,'NumNeighbors',4,'Radius',4);
    nSize = length(featureVector);
    nTrainData = size(imgDataTrain, 2);
    featuresDataTrain = zeros(nSize, nTrainData);
    for i = 1:nTrainData
        imgI1D = imgDataTrain(:,i);
        imgI2D = reshape(imgI1D,28,28);
        featuresDataTrain(:,i) = extractLBPFeatures(imgI2D,'NumNeighbors',4,'Radius',4);
    end
end