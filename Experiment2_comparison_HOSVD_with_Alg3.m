clear all, close all

% You can download the required data from 
% https://doi.org/10.4121/uuid:cb37d1b8-a505-46eb-8c42-fe819429624b 
load('Experiment2_top_tensor.mat')
figure,imshow(uint8(A))

Rprime=size(A,1);
I=19;

% Construct isometry and apply to the top tensor
W=orth(randn(I^2,Rprime));
B=W*A*W';
figure,imshow(uint8(B))

% Construct disentangler and apply
V=orth(randn(I^2,I^2));
C=V*reshape(permute(reshape(B,I*ones(1,4)),[2,3,4,1]),[I^2,I^2]);
C=reshape(permute(reshape(C,I*ones(1,4)),[4,1,2,3]),[I^2,I^2]);
figure,imshow(uint8(C))

c=Core(reshape(C,I*ones(1,4)));
[mps,e]=c.mps(1e-16)


% Algorithm 3 to compute disentangler to cover core 2 and 3,  with low-rank, use first 128 rank-1 terms in the orthogonal Procrustes problem.
% Stopping condition is rank-gap is 1e15 or maximum 50000 iterations have
% passed.
% Since the original rank is 361 and has to go down to 128, this will take a while.
[V_alg3,svals,procruste]=disentangle(mps,2,mps.normcore,1:128,1e14,25000); 
figure
semilogy(svals')

% large drop in singular values between 128 and 129, which means the
% disentangler that was recovered correctly retrieves a rank-128 solution.
svals(129,end)/svals(128,end)

% Algorithm 2 without Algorithm 3
[mera_alg2,e]=openMERA(mps,2,1e-15)

% recovered isometry has output dimension of 361 instead of 128, which
% means the recovered disentangler does not recover the rank-128 solution.
mera_alg2.layers{1}.W{1}


