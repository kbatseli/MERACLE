clear all, close all

% You can download the required data from 
% https://doi.org/10.4121/uuid:cb37d1b8-a505-46eb-8c42-fe819429624b 

% load tensor with simulation data
load('Experiment1_Heatsimulation.mat')

% convert tensor to Core object
A=Core(T)

% convert tensor A into an MPS with maximal relative approximation error of
% 1e-3
tic;
[mps,e]=A.mps(1e-3)
toc
% compression ratio
round(A.numel/mps.numel)

% convert the obtained TT into a HOSVD
tic;
[Smps,U,e]=mps.THOSVD(1e-13)
toc
% compression ratio
round(A.numel/(Smps.numel+sum(cellfun(@(x)numel(x),U))))

% perform same experiment on 16-way reshaped tensor
A=Core(reshape(T,[factor(size(T,1)),factor(size(T,2)),factor(size(T,3))]))
tic;
[mps,e]=A.mps(1e-3)
toc

% compression ratio
round(A.numel/mps.numel)

% convert the obtained TT into a HOSVD
tic;
[Smps,U,e]=mps.THOSVD(1e-13)
toc
round(A.numel/(Smps.numel+sum(cellfun(@(x)numel(x),U))))

