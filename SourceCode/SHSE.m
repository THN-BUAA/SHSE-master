function perf = SHSE(trainData, testData, learner, locIndex, numLearners, feaRatio)
% SHSE Summary of this function goes here
%   Detailed explanation goes here
% INPUTS:
%   (1) trainData - n1_instances*(n_features+1) where the last column is the number of defects. 
%   (2) testData -  n2_instances*(n_features+1) where the last column is the number of defects.
%   (3) learner - a string, {'LR','CART','RF'}.
%   (4) locIndex - the index of LOC.
%   (5) numLearners {1,2,...} - the number of learners.
%   (6) feaRatio (0,1] - how much features are selected.
% OUTPUTS:

% Default values
if ~exist('numLearners','var')||isempty(numLearners)
    numLearners = 50;
end

if ~exist('feaRatio','var')||isempty(feaRatio)
    feaRatio = 3/4; % 
end


insRatio = 0;
if strcmp(learner, 'LR')||strcmp(learner, 'CART')||strcmp(learner, 'RF')
    ratioDef = sum(trainData(:,end)>0)/size(trainData,1);
    if ratioDef<0.5 % 
        if ratioDef+ratioDef*0.5 <= 0.5
            insRatio = (ratioDef+ratioDef*0.5)/(1-ratioDef);
        else
            insRatio = 0.5/(1-ratioDef);
        end
    else
        insRatio = 1;
    end
end

trainXOri = trainData(:,1:(end-1));
trainYOri = trainData(:,end);
trainLOC = trainData(:,locIndex);

testXOri = testData(:,1:(end-1));
testYOri = testData(:,end);
testLOC = testData(:,locIndex);

K = numLearners;
selFeaNum = floor(size(trainXOri,2)*feaRatio);
modelCell = cell(1,K);
weight = zeros(2,K);
preES = zeros(size(testXOri,1),K);

idxPos = (trainYOri>0);
trainXPos = trainXOri(idxPos,:);
trainYPos = trainYOri(idxPos,:);

trainXNeg = trainXOri(~idxPos,:);
trainYNeg = trainYOri(~idxPos,:);


for j0=1:K
    
    rand('seed',j0);
    idxSelFea = randperm(size(trainXOri, 2), selFeaNum);
    
    idxSelInsPosNum = size(trainXPos,1);
    idxSelInsNegNum = floor(size(trainXNeg,1)*insRatio); 
    
    
    idxPos = randperm(size(trainXPos,1),idxSelInsPosNum);
    idxNeg = randperm(size(trainXNeg,1),idxSelInsNegNum);
    
    newTrainX = [trainXPos(idxPos,idxSelFea);trainXNeg(idxNeg,idxSelFea)];
    newTrainY = [trainYPos(idxPos,:);trainYNeg(idxNeg,:)];
    
    matData = unique([newTrainX,newTrainY],'rows','stable');
    newTrainX = matData(:,1:(end-1));
    newTrainY = matData(:,end);
    
    % SMOTEND
    synMino = SMOTEND([newTrainX newTrainY]); 
    
    if ~isempty(synMino)
        newTrainXBal = [newTrainX; synMino(:,1:(end-1))];
        newTrainYBal = [newTrainY; synMino(:,end)];
    else
        newTrainXBal = newTrainX;
        newTrainYBal = newTrainY;
    end
    
    
    % Fitting
    switch learner
        case 'LR'
            mdl = fitlm(newTrainXBal, newTrainYBal);
            fitY = predict(mdl, trainXOri(:,idxSelFea)); %
        case 'CART'
            mdl = fitrtree(newTrainXBal,newTrainYBal);
            fitY = predict(mdl, trainXOri(:,idxSelFea)); %
        case 'RF'
            nTrees = 20;
            mdl = TreeBagger(nTrees, newTrainXBal, newTrainYBal, 'Method','regression');
            fitY = predict(mdl, trainXOri(:,idxSelFea)); %
    end

    fitY = round(fitY); 
    
    fitY(fitY<0) = 0;
    tempTr = RegPerformance(trainYOri, fitY, trainLOC);
    weight(1,j0) = tempTr.rmse; 
    weight(2,j0) = tempTr.fpa;  
    modelCell{j0} = mdl;
    
    % Prediction
    preDes = predict(mdl, testXOri(:,idxSelFea));
    preNum = round(preDes);
    
    preNum(preNum<0) = 0;
    preES(:,j0) = preNum; 
end


weight(1,:) = sum(weight(1,:))./(weight(1,:)+eps); 
weight(1,:) = weight(1,:)/(sum(weight(1,:))+eps);
preE = round(preES*weight(1,:)');


perf = FPA(testYOri, preE);

end

