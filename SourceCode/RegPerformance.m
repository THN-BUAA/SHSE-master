function outputs = RegPerformance(actY, preY, LOC)
%REGPERFORMANCE Summary of this function goes here
%   Detailed explanation goes here
% Inputs:
%   (1) actY - n*1 vector
%   (2) preY - n*1 vector
% Outputs:
%   outputs
%

if~exist('LOC','var')||isempty(LOC)
    LOC = 0;
end
outputs.fpa = FPA(actY, preY);
outputs.kendall = corr(actY, preY, 'type' , 'kendall');
outputs.rmse = sqrt(mean((preY-actY).^2));

end

