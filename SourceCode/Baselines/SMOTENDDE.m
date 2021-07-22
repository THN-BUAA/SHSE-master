function balancedSource = SMOTENDDE(source,learner, P, D, T, F, CR)
%SMOTENDDE Summary of this function goes here£º Differential Evolutionary for SMOTEND
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

%% Default value
if ~exist('learner','var')||isempty(learner)
    learner = 'CART'; 
end
if ~exist('P','var')||isempty(P)
    P = 30;  % 30
end
if ~exist('D','var')||isempty(D)
    D = 3; 
end
if ~exist('T','var')||isempty(T)
    T = 8; % 8
end
if ~exist('F','var')||isempty(F)
    F = 0.7; 
end
if ~exist('CR','var')||isempty(CR)
    CR = 0.3; 
end

% Value range
k_min = 1; k_max = 20;
m_min = 0; m_max = 6; 
r_min = 0.2; r_max = 5;

% Initialization values of best parameters
best_k = 5;
best_m = 6;
best_r = 2;

%% Divided into training dataset and validation dataset
posSrc = source(source(:,end)>0,:);
negSrc = source(source(:,end)==0,:);  

valPosIdx = randperm(size(posSrc,1), floor(0.25*size(posSrc,1)));
valNegIdx = randperm(size(negSrc,1), floor(0.25*size(negSrc,1)));

trainData = [posSrc(setdiff(1:size(posSrc,1),valPosIdx), :); negSrc(setdiff(1:size(negSrc,1),valNegIdx), :)];
validData = [posSrc(valPosIdx, :); negSrc(valNegIdx, :)];


%%
X = [];
for i=1:P % each population  
    %
    k = k_min + round(rand * (k_max - k_min));
    m = m_min + round(rand * (m_max - m_min));
    r = r_min + rand * (r_max - r_min);
    
    X = [X, [k,m,r]'];
    %  
    if funFitness(best_k, best_m, best_r, trainData, validData, learner) < funFitness(k, m, r, trainData, validData, learner)
        best_k = k;
        best_m = m;
        best_r = r;
    end 
end

ranges.k_min = k_min;
ranges.k_max = k_max;
ranges.m_min = m_min;
ranges.m_max = m_max;
ranges.r_min = r_min;
ranges.r_max = r_max;
%% Population evolution
for i=1:T % each iteration
    for j=1:P % each population
        v = getNewInstance(D,F,CR,X(:,j),X, ranges);
        if funFitness(best_k, best_m, best_r, trainData, validData, learner) < funFitness(v(1), v(2), v(3), trainData, validData, learner)
            best_k = v(1);
            best_m = v(2);
            best_r = v(3);
        end
    end
end

%% Balance given data with the best parameters
balancedSource = synGenerate(best_k, best_m, best_r, trainData);

end



function  v = getNewInstance(D, F, CR, target, X, ranges)
%GETNEWINSTANCE Summary of this function goes here: Implement Algorithm 3 in [1]
%   Detailed explanation goes here
% INPUTS:
%
% OUTPUTS:
%   v - a column vector with size D

%% Performing mutation operator
selThreeX = X(:, randperm(size(X,1),3)); % Randomly select three columns  
u = selThreeX(:,1) + F * (selThreeX(:,2) - selThreeX(:,3)); 
u = [round(u(1)),round(u(2)),u(3)];

% Must ensure parameter value falls within the specified range
u(1) = min(u(1), ranges.k_max); u(1) = max(u(1), ranges.k_min);
u(2) = min(u(2), ranges.m_max); u(2) = max(u(2), ranges.m_min);
u(3) = min(u(3), ranges.r_max); u(3) = max(u(3), ranges.r_min);

%% Performing crossover operator
v = zeros(D,1); % Initialization
rd = randi(D);  % Return an integer between 1 and D 
for k=1:D 
    if rand<CR || k==rd
        v(k) = u(k);
    else
        v(k) = target(k);
    end
end

end


function fitness = funFitness(k, m, r, train, validation, learner)
%FUNFITNESS Summary of this function goes here
%   Detailed explanation goes here
% INPUTS:
%   (1) k
%   (2) m
%   (3) r
%   (4) train - a n*(d+1) matrix where the last column is the number of defects
%   (5) validation - a m*(d+1) matrix
%   (6) learner - a string to denote the regression technique, such as CART, SVR, RF. 
% OUTPUTS:
%   fitness - FPA, a real number
%

newTrain = synGenerate(k, m, r, train); % self-defined fucntion

newTrainXBal = newTrain(:,1:end-1); newTrainYBal = newTrain(:,end);

switch learner
    case 'LR'
        mdl = fitlm(newTrainXBal, newTrainYBal);
        predY = predict(mdl, validation(:,1:end-1)); %
    case 'CART'
        mdl = fitrtree(newTrainXBal,newTrainYBal);
        predY = predict(mdl, validation(:,1:end-1)); %
    case 'RF'
        nTrees = 20;
        mdl = TreeBagger(nTrees, newTrainXBal, newTrainYBal, 'Method','regression');
        predY = predict(mdl, validation(:,1:end-1)); %
end

predY = round(predY);
predY(predY<0) = 0;
fitness = FPA(validation(:,end), predY);

end


function balancedData = synGenerate(k, m, r, Data, seed)
%SYNGENERATE Summary of this function goes here: Implement Algorithm 1 in [1]
%   Detailed explanation goes here
% INPUTS:
%   (1)
%   (2)
% OUTPUTS:
%   balancedData - 
%


if ~exist('seed','var')||isempty(seed)
    seed = 1; 
end

rand('seed', seed); % For reproduction

N = 6;  % Used by Chen et al.
posTrain = Data(Data(:,end)>0, :);
negTrain = Data(Data(:,end)==0, :);
numSythetic = round((size(negTrain,1)-size(posTrain,1))*m/N);

[~,kNearIdx] = pdist2(posTrain,posTrain,'minkowski',r,'Smallest',k+1); % kNearIdx is a (k+1)*size(posTrain,1) matrix

kNearIdx(1,:) = []; % remove the first row, because these are themselves' index

index = 1;
synPos = [];
while numSythetic
    v1 = posTrain(index,:);
    v2 = posTrain(kNearIdx(randi(min(k, size(posTrain,1)-1)),index), :); % randi(nMax) - return an random integer between 1 and nMax; 
    
    newX = v1(1:end-1) + rand*(v2(1:end-1)-v1(1:end-1));
    d1 = pdist([newX;v1(1:end-1)], 'minkowski',r);
    d2 = pdist([newX;v2(1:end-1)], 'minkowski',r);
    newY = round((d1*v2(end)+d2*v1(end))/(d1+d2));
    synPos = [synPos; [newX, newY]];
    index = max(mod(size(posTrain,1), (index + 1)), 1); % Ensure index >= 1
    numSythetic = numSythetic - 1;
end
balancedData = [Data; synPos];
end

