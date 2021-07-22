function perf = SMOTENDDE_model(train, test, learner, testLOC)
%SMOTENDDE_MODEL Summary of this function goes here£º Differential Evolutionary for SMOTEND
%   Detailed explanation goes here
% INPUTS:
%   (1) source -  a n*(d+1) matrix where n and d denote the number of
%    samples and number of features, the last column is the # of defects.
%   (2) P - Population Size
%   (3) D - Number of Parameters
%   (4) T - Number of Generations
%   (5) F - Differential Weight 
%   (6) CR - Crossover Probability
% OUTPUTS:
%   (1) best_k
%   (2) best_m
%   (3) best_r
%
%
% Reference: [1] Chen, X., et al., Software defect number prediction:
%            Unsupervised vs supervised methods. Information and Software Technology,
%            2019. 106: p. 161-181. DOI: 10.1016/j.infsof.2018.10.003.
%            [2] A. Agrawal, T. Menzies, Is "better data" better than
%            "better data miners"? (on the benefits of tuning smote for
%            defect prediction), in: Proceedings of the International Conference 
%            on Software Engineering, 2018, pp. 1050¨C1061.

balancedTrain = train;
if sum(train(:,end)>0) < sum(train(:,end)==0)
    balancedTrain = SMOTENDDE(train, learner); % Call for self-defined function - SMOTENDDE
end
balTrainX = balancedTrain(:,1:end-1);
balTrainY = balancedTrain(:,end);
switch learner
    case 'LR'
        mdl = fitlm(balTrainX, balTrainY);
        predY = predict(mdl, test(:,1:end-1)); %
    case 'CART'
        mdl = fitrtree(balTrainX,balTrainY);
        predY = predict(mdl, test(:,1:end-1)); %
    case 'RF'
        nTrees = 20;
        mdl = TreeBagger(nTrees, balTrainX, balTrainY, 'Method','regression');
        predY = predict(mdl, test(:,1:end-1)); %
end

% Ensure Y is non-negative integer
predY = round(predY);
predY(predY<0) = 0;

% Prediction
perf = FPA(test(:,end), predY, testLOC);
end

