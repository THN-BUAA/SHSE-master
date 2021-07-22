function SynSamples = SMOTEND(dataSet, ideRatio, k)
%SMOTEND Summary of this function goes here
%   Detailed explanation goes here
% Inputs:
%   (1) dataSet - a n_samples*(d+1) matrix where d is the numeber of defects.
%   (2) ideRatio - ideal defective ratio
%   (3) k - the number of nearest neighbors
% Outputs
%   SynSamples
%
% Reference: X. Yu, J. Liu, Z. Yang, X. Jia, Q. Ling, S. Ye, Learning from
% imbalanced data for predicting the number of software defects, in: 28th
% IEEE International Symposium on Software Reliability Engineering, ISSRE
% 2017, IEEE Computer Society, Toulouse, France, 2017, pp.
% 78¨C89.doi:10.1109/ISSRE.2017.18.


if nargin<1
    error('Please check the parameters!');
end
if ~exist('ideRatio','var')||isempty(ideRatio)
    ideRatio = 1;
end
if ~exist('k','var')||isempty(k)
    k = 5;
end


rand('seed',1);

n = size(dataSet,1); % Number of samples
% m = sum(dataSet(:,end)==1); % Number of minority class samples
% m = sum(~dataSet(:,end)==0); % Number of minority class samples
m = sum(dataSet(:,end)~=0); % Number of minority class samples


if m>=(n-m) 
    SynSamples = [];
    return;
end

minoSam = dataSet(dataSet(:,end)~=0,:); %
majoSam = dataSet(dataSet(:,end)==0,:);  
minoSamX = minoSam(:,1:end-1);


gNum = (n-m)*ideRatio-m; % Number of synthetic minority class samples
if gNum < m % 
    indexOri = 1;
    m = floor(gNum);
else
    indexOri = floor(gNum/m); % 
end

D = dist(minoSamX'); 
D = D - eye(size(D,1),size(D,1)); 
[~, idx] = sort(D, 2); 
idx = idx(:,2:end); 


SynSamples = zeros(m*indexOri, size(dataSet,2)); 
count = 1; 
for i=1:m 
    index = indexOri;
    while index
        if k<=size(idx,2) % 
            nn = idx(i,randperm(k,1)); 
        else
            
            temp0 = size(idx,2);
            temp = randperm(temp0,1);
            nn = idx(i,temp,1); 
            
        end
        
        xnn = minoSamX(nn,:);
        xi = minoSamX(i,:);
        xSyn = xi + rand * (xnn - xi);
        d1 = norm(xSyn - xi); % distance between xSyn and xi
        d2 = norm(xSyn - xnn); % distance between xSyn and xnn
        ySyn = (d2*minoSam(i,end)+d1*minoSam(nn,end))/(d1+d2);

        SynSamples(count,:) = [xSyn, ySyn];
        count = count + 1;
        index = index - 1;
    end
end

SynSamples(isnan(SynSamples(:,end)),:) = []; 

end
