function [mera,e]=opMERA(mps,k,varargin)
% mera = mera=opMERA(mps,k,r)
% ---------------------------
% Initializes a MERA with open boundary conditions from a given mps through
% consecutive hoSVD operations. 
%
% mera      =   MERA object.
%
% e         =   scalar, absolute approximation error.
%
% mps       =   MPT object, contains the mps from which a mera is computed.
%
% k         =   vector: order of the isometries. The ith layer of the mera
%               has isometries of order k(i).
%
% r         =   scalar, vector or cell: outgoing dimension of isometries.
%               If r is a scalar then r is then the dimensions of all
%               isometries are determined such that the resulting mera has
%               a relative approximation error upper bounded by r. If r is
%               a vector, then all outgoing dimension in layer i are r(i).
%               If r is a cell, the r{i} contains a vector of outgoing
%               dimensions for layer i.
%
% Reference
% ---------
%
% Kim Batselier

% error checking
if sum(mps.k-ones(1,mps.N))~=0
    error('This method only works for an MPS.');
elseif mod(mps.N,prod(k))~=0
    error('Number of nodes in mps needs to be divisible by prod(k)');
% elseif sum(cell2mat(mps.n)-mps.n{1}*ones(1,mps.N)) ~=0
%     error('All free nodes of the MPS need to have equal dimension.'); 
end

% make a copy of the mps so we don't damage the original mps
mps2=mps.copy;

% number of layers in the MERA is length(k)
layers=cell(1,length(k));
e=0;
for i=1:length(k)    
    p=mps2.N/k(i);
    U=cell(1,p-1);    
    W=cell(1,p);
    
    % first compute p-1 disentanglers        
    for j=1:p-1
        % contract consecutive mps cores into supercore
        supercore=mps2.subcon(j*k(i),j*k(i)+1);
        % compute an orthogonal factor matrix for the disentangler
        [u,R]=qr(supercore.bottom2top);
%         [u,S,V]=svd(supercore.bottom2top,'econ');
        U{j}=Disentangler([mps2.n{j*k(i)},mps2.n{j*k(i)+1},mps2.n{j*k(i)},mps2.n{j*k(i)+1}],reshape(u,[mps2.n{j*k(i)},mps2.n{j*k(i)+1},mps2.n{j*k(i)},mps2.n{j*k(i)+1}]));
        % replace supercore with R factor
        temp=permute(reshape(R,[mps2.n{j*k(i)},mps2.n{j*k(i)+1},mps2.r(j*k(i)+2),mps2.r(j*k(i))]),[4,1,2,3]);
%         temp=permute(reshape(S*V',[mps2.n{j*k(i)},mps2.n{j*k(i)+1},mps2.r(j*k(i)+2),mps2.r(j*k(i))]),[4,1,2,3]);
        
        temp=Core([mps2.r(j*k(i)),mps2.r(j*k(i)+2)],[mps2.n{j*k(i)},mps2.n{j*k(i)+1}],temp);
        % split the new supercore
        temp=temp.mps(0);
        % replace the 2 cores of mps2 by R factor
        mps2.setCore(j*k(i),temp.cores{1});
        mps2.setCore(j*k(i)+1,temp.cores{2});
    end
       
    % compute p isometries and replace mps2 with smaller mps
    mpt=[];
    for j=1:p
       % contract k consecutive mps cores into supercore
       mpt=[mpt,mps2.subcon((j-1)*k(i)+1,j*k(i))];       
    end
    % convert the mpt into an mps
    mpt.mps;
    mpt.rlortho;
    n=cell2mat(mps2.n); % we need to know the dimensions to reshape the isometry correctly
    % compute the isometries from a truncated hosvd
    if ~isempty(varargin)
        if iscell(varargin{1})
            % isometrie output dimensions are not identical in a layer
            [mps2,w,er]=mpt.THOSVD(varargin{1}{i});
        elseif isscalar(varargin{1})            
%             [mps2,w,er]=mpt.THOSVD(varargin{1}/length(k));
            [mps2,w,er]=mpt.THOSVD(varargin{1}/sqrt(length(k)));
        elseif isvector(varargin{1})
            % isometrie output dimensions are not identical in a layer
            [mps2,w,er]=mpt.THOSVD(varargin{1}(i)*ones(1,mpt.N));
        end
    else
        [mps2,w,er]=mpt.THOSVD;
    end
    e=e+er^2;
    for j=1:p
       W{j}=Isometry(k(i),n((j-1)*k(i)+1:j*k(i)),size(w{j},2),reshape(w{j},[n((j-1)*k(i)+1:j*k(i)),size(w{j},2)]));
    end
    layers{i}=Layer(U,W);        
end
e=sqrt(e);
top=Core([mps2.r(1),mps2.r(end)],cell2mat(mps2.n),mps2.subcon(1,mps2.N).core);
mera=MERA(layers,top);

end