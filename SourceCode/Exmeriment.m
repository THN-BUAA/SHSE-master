%% 
clear;
mainDir = '..\Datasets\AEEEM\';
dataNames = {'Eclipse JDT Core', 'Eclipse PDE UI', 'Equinox', 'Lucene', 'Mylyn'};

% Set saving path of emperiment results
filePath = 'E:\Document\Experiments\SHSE';
if ~exist(filePath,'dir')
    mkdir(filePath);
end

% Experiment Settings 
runTimes = 30; 
folds = 5;

perfs = cell(1,numel(dataNames)); 
import weka.filters.*; 
import weka.*;
dataPath = mainDir;

for d=1:numel(dataNames) 
    
    rand('seed',d);
    
    disp(['Datasets:', num2str(d), '/', num2str(numel(dataNames))]);
    name = dataNames{d};
    
    % Load .arff file
    file = java.io.File([dataPath, name, '.arff']);
    loader = weka.core.converters.ArffLoader;  
    loader.setFile(file); 
    insts = loader.getDataSet; 
    
    % Repalce missing value with mean value
    filter = javaObject('weka.filters.unsupervised.attribute.ReplaceMissingValues');
    filter.setInputFormat(insts);
    insts = weka.filters.Filter.useFilter(insts,filter);
    
    % Transform arff to mat
    [mat0,featureNames,targetNDX,stringVals,relationName] = weka2matlab(insts,[]); 
    
    % Remove duplicated instances
    mat0 = unique(mat0,'rows','stable');
    
    % Remove Inconsistent instances
    mat0X = mat0(:,1:end-1);
    idx = [];
    for i=1:size(mat0,1)-1
        for j=i+1:size(mat0,1)
            if sum(mat0X(i,:)==mat0X(j,:))==size(mat0X,2) && mat0(i,end)~=mat0(j,end)
                idx = [idx, i,j];
            end
        end
    end
    idx = unique(idx);
    mat0(idx,:) = [];
    
    % Remove instances with loc is 0
    if size(mat0,2)==18 
        LOCIdx = 9;  % For AEEEM datasets
    else
        LOCIdx = 11; % For PROMISE datasets
    end
    mat0(mat0(:,LOCIdx)==0,:) =[];
    
    
    % Shuffle the dataset
    mat0 = mat0(randperm(size(mat0,1), size(mat0,1)),:);

    % Initialization
    perf_DT = zeros(runTimes,1); perf_DTS = zeros(runTimes,1); perf_SHSE = zeros(runTimes,1); 
    
    for i=1:runTimes
        
        rand('seed', i);
        disp([ 'runtimes: ', num2str(i), '/' ,num2str(runTimes)]);
        
        % Initialization of folds performance
        perf_DT_folds = zeros(folds,1); perf_SHSE_folds = zeros(folds,1); perf_DTS_folds = zeros(folds,1); 
        
        % K-Fold
        indices=crossvalind('Kfold', zeros(size(mat0,1),1),folds); 
        
        for j=1:folds

            % Divide training and testing datasets
            testFlag = (indices==j);
            trainXOri = mat0(~testFlag, 1:(end-1));
            trainYOri = mat0(~testFlag, end);
            trainYOriLab = double(mat0(~testFlag, end)>=1);
            trainLOC = mat0(~testFlag,LOCIdx);
            
            testXOri = mat0(testFlag, 1:(end-1));            %
            testYOri = mat0(testFlag, end);               % 
            testYOriLab = double(mat0(testFlag, end)>=1); % 
            testLOC = mat0(testFlag, LOCIdx);             % 
            
            
            if unique(testYOri)==0
                perf_DT_folds(j,:) = nan;perf_DTS_folds(j,:) = nan; perf_SHSE_folds(j,:) = nan;
                continue; 
            end
            
            
           %% Baseline: CART + NONE
            disp('CART...');
            rand('seed',0);
            trainX = trainXOri;
            tree = fitrtree(trainX,trainYOri);
            preDT = predict(tree, testXOri);  
            preDT = round(preDT);
            perf_DT_folds(j,:)=FPA(testYOri, preDT);
                       
            %% Baseline: CART + SMOTEND
            disp('SMOTEND ...');
            rand('seed',0);
            trainX = trainXOri;
            synSamS = SMOTEND([trainXOri,trainYOri]);
            if ~isempty(synSamS)
                trainX1 = [trainX;synSamS(:,1:end-1)];
                trainY1 = [trainYOri;synSamS(:,end)];
            else
                trainX1 = trainX;
                trainY1 = trainYOri;
            end
            tree_s = fitrtree(trainX1,trainY1);
            preSDT = predict(tree_s, testXOri);
            preSDT = round(preSDT);
            perf_DTS_folds(j,:) = FPA(testYOri, preSDT);
                       
            %% SHSE
            disp('SHSE ...');
            perf_SHSE_folds(j,:) = SHSE([trainXOri, trainYOri],[testXOri, testYOri],'CART', LOCIdx); 
           
            
        end % end of folds
        perf_SHSE(i,:) = mean(perf_SHSE_folds, 1); perf_DT(i,:) = mean(perf_DT_folds, 1); perf_DTS(i,:) = mean(perf_DTS_folds, 1);
       
    end % end of runs
    perfs = {perf_DT. perf_DTS, perf_SHSE};
    save([filePath,'\perfs.mat'],'perfs');
end % 
a = 1;