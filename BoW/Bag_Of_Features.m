function Bag_Of_Features()
    % Nap du lieu train
     fprintf('\nNap du lieu train ....');
    rootFolder = fullfile('./DataTrain');
    categories = {'0','1','2','3','4','5','6','7','8','9'};
    imds = imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');     
    tbl01 = countEachLabel(imds);
    minSetCount = min(tbl01{:,2});
    imds = splitEachLabel(imds,minSetCount,'randomize');   
    bag = bagOfFeatures(imds);
   
    % Xay dung model
    fprintf('\nXay dung model ....');    
    categoryClassifier = trainImageCategoryClassfier(imds,bag);
    
    % Nap du lieu test
    fprintf('\nNap du lieu test ....');
    rootFolder = fullfile('./DataTest');
    categories = {'0','1','2','3','4','5','6','7','8','9'};
    imds = imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');   
    
    % Predict
    fprintf('\nPredict ....');
    confMatrixTest = evaluate(categoryClassifier, imds);
    mean(diag(confMatrixTest));    
end