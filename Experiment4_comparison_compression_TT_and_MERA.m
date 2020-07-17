clear all, close all

% Construct a 2-layer MERA of a 12-D cubical tensor where each mode has
% size 10
n=10;
N=12;
k=[2,2]; % top is 3-way
ranks=[5,5];
nin=[n,ranks(1)];

layers=cell(1,length(k));
for j=1:length(k)
    p=N/prod(k(1:j));
    W=cell(1,p);
    U=cell(1,p-1);
    for i=1:p-1
        U{i}=Disentangler(nin(j)*ones(1,4),reshape(orth(randn(nin(j)^2,nin(j)^2)),nin(j)*ones(1,4)));
    end
    for i=1:p
        W{i}=Isometry(k(1),nin(j)*ones(1,2),ranks(j),reshape(orth(randn(nin(j)^2,ranks(j))),[nin(j),nin(j),ranks(j)]));
    end
    layers{j}=Layer(U,W);
end
top=Core([1,1],ranks(end)*ones(1,N/prod(k)),randn([ranks(end)*ones(1,N/prod(k)),1]));
mera=MERA(layers,top);
clear U W 

% compression ratio
round(10^12/mera.numel)

% construct the corresponding Tensor Train from the MERA with approximation
% error of zero.
[mps,e]=mera.mps(1e-15)

% compression ratio
round(10^12/mps.numel)

%% Use Algorithm 2 (consecutive HOSVD/Truncated HOSVD computations) to convert Tensor Train into a MERA
tic;
[mera_alg2,e]=openMERA(mps,k,ranks),
toc
% relative error e is 100% due to faulty disentanglers

%% Use Algorithm 2 together with Algorithm 3 to compute first layer of the MERA 
term=[5,25,25,25,5];
MAXITR=1000;
MERATime=zeros(1,10);
for i=1:5
    from=2*i;
    to=from+1;
    v=tic;
    [U{i},svals,procruste]=disentangle(mps,from,from+2,1:term(i),1e13,MAXITR); % Algorithm 3 to compute disentanglers with low-rank
    MERATime(i)=toc(v);
end

% Apply computed disentanglers on the original mps
N=mps.N;
k=mera.layers{1}.k;
p=N/k;
new_mps=mps.submpt(1,k-1);
for i=1:p-1
    from=2*i;
    to=from+1;
    % 2 neighbouring nodes of the MPS are contracted
    temp=mps.subcon(from,to);
    % contract the MPS supercore with the disentangler
    temp=Core([mps.r(from),mps.r(to+1)],[mps.n{from},mps.n{to}],permute(reshape(U{i}*temp.top2bottom',[mps.n{from},mps.n{to},mps.r(from),mps.r(to+1)]),[3,1,2,4]));
    
    % split temp supercore into 2 linked cores with svd
    split_temp=temp.mps(1e-10);    
    % concatenate 2 cores with remaining k-2 ones
    if k==2
        new_mps=[new_mps,split_temp];
    else
        new_mps=[new_mps,split_temp,mps.submpt(i*k+2,(i+1)*k-1)];
    end
end
new_mps=[new_mps,mps.submpt(N,N)];

% sum over alternating ranks to compute isometries
test=[];
for i=1:p
    % contract k nodes of the new mps together 
    test=[test,new_mps.subcon((i-1)*k+1,i*k)];
end
test.mps;
v=tic;
[Smps,W,e1]=THOSVD(test,1e-12); % Truncated HOSVD to compute the isometries
MERATime(6)=toc(v);

%% Use Algorithm 2 together with Algorithm 3 to compute second layer of the MERA 
term=5*ones(1,2);
MAXITR=1000;
for i=1:2    
    from=2*i;
    to=from+1;
    v=tic;
    [U{i},svals,procruste]=disentangle(Smps,from,from+2,1:term(i),1e13,MAXITR);  % Algorithm 3 to compute disentanglers with low-rank
    MERATime(6+i)=toc(v);
end

% Apply disentanglers on the mps
N=Smps.N;
k=mera.layers{1}.k;
p=N/k;
new_mps=Smps.submpt(1,k-1);
for i=1:p-1
    from=2*i;
    to=from+1;
    % 2 neighbouring nodes of the MPS are contracted
    temp=Smps.subcon(from,to);
    % contract the MPS supercore with the disentangler
    temp=Core([Smps.r(from),Smps.r(to+1)],[Smps.n{from},Smps.n{to}],permute(reshape(U{i}*temp.top2bottom',[Smps.n{from},Smps.n{to},Smps.r(from),Smps.r(to+1)]),[3,1,2,4]));
    
    % split temp supercore into 2 linked cores with svd
    split_temp=temp.mps(1e-10);    
    % concatenate 2 cores with remaining k-2 ones
    if k==2
        new_mps=[new_mps,split_temp];
    else
        new_mps=[new_mps,split_temp,mps.submpt(i*k+2,(i+1)*k-1)];
    end
end
new_mps=[new_mps,Smps.submpt(N,N)];

% sum over alternating ranks to compute isometries
test=[];
for i=1:p
    % contract k nodes of the new mps together 
    test=[test,new_mps.subcon((i-1)*k+1,i*k)];
end
test.mps;
v=tic;
[Smps,W,e2]=THOSVD(test,1e-12); % Truncated HOSVD to compute the isometries
MERATime(9)=toc(v);

% Total time to compute the MERA
sum(MERATime)

% Frobenius norm comparison
norm(mera.top.norm-Smps.norm)/mera.top.norm % Algorithm 2 with Algorithm 3





