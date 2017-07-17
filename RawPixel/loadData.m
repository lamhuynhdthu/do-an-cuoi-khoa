% Nap du lieu chu so viet tay
function [imgData, lblDataLabel] = loadData(strFileNameData, strFileNameLabel)
    imgData = loadMNISTImages(strFileNameData);
    lblDataLabel = loadMNISTLabels(strFileNameLabel);
end