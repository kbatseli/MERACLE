classdef MPT < handle
    %
    % Matrix Product Tensor, a linear chain of core tensors. Special cases
    % are:
    %
    % Matrix Product State (MPS), also called Tensor Train: all cores are
    % 3-way,
    %
    % Matrix Product Operator (MPO): all cores are 4-way.
    %
    % Not all cores need to have the same amount of free indices. When they
    % do, then they are arranged as first mode, second mode, etc... of the
    % resulting tensor.
    %
    %
    %           -------       -------                    -------
    %      r(1) |     |  r(2) |     |  r(3)         r(N) |     |  r(N+1)
    %     ------|  1  |-------|  2  |------- .... -------|  N  |--------   
    %           |     |       |     |                    |     |
    %           -------       -------                    -------   
    %           | ... |       | ... |                    | ... |        
    %           
        properties (SetAccess = private)
        r           % vector of N ranks, as indicated in above diagram
        n           % n{i}= vector with dimensions of bottom legs in ith core
        k           % k(i) is the number of bottom legs on ith core
        cores       % core{i} is the ith Core object
        N           % number of cores in the chain
        normcore    % integer, core that contains the norm of the whole tensor, can be zero
    end
    
    methods
        % constructor
        % call either MPT(cores) or MPT(cores,normcore)
        % where cores is a cell of Core objects and normcore is integer
        % that denotes the core that contains the whole Frobenius norm.
        function mpt=MPT(lcores,varargin)
            if ~iscell(lcores)
                lcores={lcores};
            end
            mpt.cores=lcores;
            mpt.N=length(lcores);
            for i=1:length(lcores)
                mpt.r(i)=lcores{i}.r(1);
                mpt.n{i}=lcores{i}.n;
                mpt.k(i)=lcores{i}.k;
            end
            mpt.r(length(lcores)+1)=lcores{length(lcores)}.r(2);
            if ~isempty(varargin)
                mpt.normcore=varargin{1};
            else
                mpt.normcore=0;
            end
        end
                
        
        %%   
        % ------------------        
        % |  CONTRACTIONS  |
        % ------------------  
        
        function tensor=contract(mpt)
            % contracts the MPT into a tensor, use this only if all the
            % linking ranks match, including the first and last.
            if mpt.N==1
                tensor=mpt.cores{1}.contract;
            else
                tensor=mpt.cores{1};
                for i=2:mpt.N-1
                    tensor=tensor.rightcon(mpt.cores{i});
                end
                tensor=tensor.topcon(mpt.cores{mpt.N});
                % check whether all k's are equal
                if sum(abs(mpt.k-mpt.k(1)*ones(1,mpt.N))) == 0
                    % all k's are equal, need to reshape and permute the
                    % corresponding tensor
                    k=mpt.k(1);
                    n=cell2mat(mpt.n);
                    tensor=reshape(tensor.core,n);                    
                    permvec=reshape(1:k*mpt.N,[k,mpt.N])';
                    n=reshape(n(permvec(:)'),[mpt.N,k]);
                    if size(n,2)==1
                        n=[n ones(mpt.N,1)];
                    end
                    tensor=reshape(permute(tensor,permvec(:)'),prod(n,1));
                else
                    tensor=squeeze(tensor.core);
                end
            end
        end
        
        function core=subcon(mpt,from,to)
           % contracts cores in the MPT starting at core "from" and going
           % to the right up to and including core "to", wraps around!
           core=mpt.cores{from};
           if to < from
               indices=[from+1:mpt.N 1:to];
           else
               indices=from+1:to;
           end
           for i=1:length(indices)
               core=core.rightcon(mpt.cores{indices(i)});
           end
        end
        
        function core=topbottomcon(mpt,mpt2)
           % contracts mpt with mpt2 over all their free indices, resulting
           % in either a matrix, vector or scalar. Obviously, all cores
           % need to have matching number of free indices and dimensions
           for i=1:mpt.N
              cores{i}=mpt.cores{i}.topbottomcon(mpt2.cores{i});
           end
           result=MPT(cores);
           core=result.subcon(1,mpt.N);
           core.squeeze;
        end

        function mpt=modecon(mpt1,k1,mpt2,k2)
           % contracts mpt1 along mode k1 with mpt2 along mode k2
           cores=cell(1,mpt1.N);
           for i=1:mpt1.N
               cores{i}=mpt1.cores{i}.modekcon(k1,mpt2.cores{i},k2);
           end
           mpt=MPT(cores);
        end
        
        function n=norm(mpt)
            % Computes Frobenius norm of tensor in MPT form
            if mpt.normcore ~= 0
                n=norm(mpt.cores{mpt.normcore}.core(:));
            else                
                mpt.rlortho; % put norm in first core
                n=norm(mpt.cores{1}.core(:));               
            end
        end      
        
        function rdm=rdm(mps,index1,index2)
            % computes the reduced density matrix from a given mps over
            % indices index1 and index2
            if ~isempty(find(mps.k>1))
                error('This method only works for an MPS');
            end
            temp=sort([index1 index2]);
            index1=temp(1);
            index2=temp(2);
            left=mps.submpt(1,index1-1);
            if index2==index1+1
                center=[];
            else
                center=mps.submpt(index1+1,index2-1);
            end
            right=mps.submpt(index2+1,mps.N);
            core1=mps.cores{index1}.kron(mps.cores{index1});
            core2=mps.cores{index2}.kron(mps.cores{index2});
            if ~isempty(left)
                left=left.topbottomcon(left);
            end
            if ~isempty(center)
                center=center.topbottomcon(center);
            end
            if ~isempty(right)
                right=right.topbottomcon(right);
            end
            mpt2=[left core1 center core2 right];
            rdm=reshape(permute(reshape(mpt2.contract,[mps.n{index1}*ones(1,2),mps.n{index2}*ones(1,2)]),[1 3 2 4]),[cell2mat(mps.n(index1))*cell2mat(mps.n(index1)),cell2mat(mps.n(index1))*cell2mat(mps.n(index1))]); 
        end       

        %%   
        % ----------------------------        
        % |   OVERLOADED OPERATORS   |
        % ----------------------------    
        
        function mpt=horzcat(varargin)
            % mpt=horzcat(mpt1,mpt2,...,mptn)
            % concatenates mpt1 with mpt2 to obtain mpt1--mpt2--..--mptn, 
            % all links should have matching dimensions
            % mpt=vertcat(mpt1,mpt2,...,mptn)            
            % concatenates mpt1 with mpt2 to obtain mpt1--mpt2--..--mptn, 
            % all links should have matching dimensions
            varargin=varargin(~cellfun('isempty',varargin));
            if isa(varargin{1},'Core')
                mpt=MPT(varargin{1});
            else
                mpt=varargin{1};
            end
            for i=2:length(varargin)
                if isa(varargin{i},'Core')
                    if mpt.r(end) ~=varargin{i}.r(1)
                        error('Dimension of the link does not match')
                    else
                        mpt=[mpt MPT(varargin{i})];
                    end
                else
                    if mpt.r(end) ~= varargin{i}.r(1)
                        error('Dimension of the link does not match')
                    else
                        mpt=MPT([mpt.cores,varargin{i}.cores]);
                    end
                end
            end
        end
        
        function mpt=vertcat(varargin)
            % mpt=vertcat(mpt1,mpt2,...,mptn)            
            % concatenates mpt1 with mpt2 to obtain mpt1--mpt2--..--mptn, 
            % all links should have matching dimensions
            
%             mpt=horzcat(varargin);    %% Why doesn't this work??
            varargin=varargin(~cellfun('isempty',varargin));
            if isa(varargin{1},'Core')
                mpt=MPT(varargin{1});
            else
                mpt=varargin{1};
            end
            for i=2:length(varargin)
                if isa(varargin{i},'Core')
                    if mpt.r(end) ~=varargin{i}.r(1)
                        error('Dimension of the link does not match')
                    else
                        mpt=[mpt MPT(varargin{i})];
                    end
                else
                    if mpt.r(end) ~= varargin{i}.r(1)
                        error('Dimension of the link does not match')
                    else
                        mpt=MPT([mpt.cores,varargin{i}.cores]);
                    end
                end
            end
        end
        function mpt=uminus(mpt1)
            mpt=MPT(mpt1.cores);
            mpt.setCore(1,Core(mpt.cores{1}.r,mpt.cores{1}.n,-mpt.cores{1}.core));
        end        
        
        function mpt=minus(mpt1,mpt2)
            if isa(mpt1,'MPT') && isa(mpt2,'MPT')
              mpt=mpt1+(-mpt2);                 
            else
                error('One of the provided arguments was not an MPT.');
            end
        end
        
        function mpt=mtimes(mpt1,a)
            % multiplication of mpt with scalar
            if (isa(mpt1,'MPT') && isscalar(a))
                mpt=MPT(mpt1.cores);
                mpt.setCore(1,Core(mpt.cores{1}.r,mpt.cores{1}.n,a*mpt.cores{1}.core));
            elseif (isa(a,'MPT') && isscalar(mpt1))
                mpt=MPT(a.cores);
                mpt.setCore(1,Core(a.cores{1}.r,a.cores{1}.n,mpt1*mpt.cores{1}.core));                
            end
        end
        
        function mpt=times(mpt1,mpt2)
           % Hadamard product of 2 tensors in identical MPT form 
           if mpt1.N~= mpt2.N || sum(mpt1.k-mpt2.k) ~= 0 || sum(cell2mat(mpt1.n)-cell2mat(mpt2.n)) ~= 0
               % two tensors have different MPT forms
               error('Two MPTs should have the same form.');
           else
               cores=cell(1,mpt1.N);
               for i=1:mpt1.N
                   cores{i}=Core([mpt1.r(i)*mpt2.r(i),mpt1.r(i+1)*mpt2.r(i+1)],mpt1.n{i},permute(reshape(khatri(mpt1.cores{i}.top2bottom,mpt2.cores{i}.top2bottom),[mpt2.r(i),mpt2.r(i+1),mpt1.r(i),mpt1.r(i+1),mpt1.n{i}]),[1,3,5:4+mpt1.k(i),2,4]));
               end
               mpt=MPT(cores);
           end
        end
        
%         function mpt2=subsref(mpt,S)
%             if S(1).type=='.'
%                 % we want a property from the object itself
%                 
%             else
%                % we want a submpt
%                mpt2=MPT(mpt.cores(S(1).subs{1}));
%             end
%         end
        
        function mpt2=submpt(mpt,from,to)
           % returns an mpt that starts at core "from" up to core "to",
           % wraps around.
           if from==0 || to==0 || from>mpt.N || to >mpt.N
               mpt2=[];
           else               
               if to < from
                   indices=[from:mpt.N 1:to];
               else
                   indices=from:to;
               end
               mpt2=MPT(mpt.cores(indices));
           end
        end
        
        
        function mpt=plus(mpt1,mpt2)
            % alternative addition according to Mickelin and Karaman
            cores=cell(1,mpt1.N);
            if mpt1.r(1) >= mpt2.r(1)
                cores{1}=Core([mpt1.r(1),mpt1.r(2)+mpt2.r(2)],mpt1.n{1},reshape([mpt1.cores{1}.right2left,[mpt2.cores{1}.right2left;zeros(mpt1.r(1)-mpt2.r(1),mpt2.r(2))]],[mpt1.r(1),mpt1.n{1},mpt1.r(2)+mpt2.r(2)]));
                cores{mpt1.N}=Core([mpt1.r(mpt1.N)+mpt2.r(mpt2.N),mpt1.r(mpt1.N+1)],mpt1.n{mpt1.N},reshape([mpt1.cores{mpt1.N}.left2right;[mpt2.cores{mpt2.N}.left2right,zeros(mpt2.r(mpt2.N),mpt1.r(mpt2.N+1)-mpt2.r(mpt2.N+1))]],[mpt1.r(mpt1.N)+mpt2.r(mpt2.N),mpt1.n{mpt1.N},mpt1.r(mpt1.N+1)]));
            else
                cores{1}=Core([mpt2.r(1),mpt1.r(2)+mpt2.r(2)],mpt1.n{1},reshape([[mpt1.cores{1}.right2left;zeros(mpt2.r(1)-mpt1.r(1),mpt1.r(2))],mpt2.cores{1}.right2left],[mpt1.r(1),mpt1.n{1},mpt1.r(2)+mpt2.r(2)]));
                cores{mpt1.N}=Core([mpt1.r(mpt1.N)+mpt2.r(mpt2.N),mpt1.r(mpt1.N+1)],mpt1.n{mpt1.N},reshape([[mpt1.cores{mpt2.N}.left2right,zeros(mpt1.r(mpt2.N),mpt2.r(mpt2.N+1)-mpt1.r(mpt2.N+1))];mpt2.cores{mpt1.N}.left2right],[mpt1.r(mpt1.N)+mpt2.r(mpt2.N),mpt1.n{mpt1.N},mpt1.r(mpt1.N+1)]));
            end
            for i=2:mpt1.N-1
                cores{i}=mpt1.cores{i}+mpt2.cores{i};
            end
            mpt=MPT(cores);
            
%            % Adds 2 MPTs together, dimension of corresponding free indices
%            % need to be equal
%            cores=cell(1,mpt1.N);
%            % first check whether both mpts share a common link dimension           
%            if ~isempty(find(mpt1.r-mpt2.r==0))
%                I=find(mpt1.r-mpt2.r==0);
% %                [minr,I2]=min(mpt1.r(I));                              
%                [minr,I2]=max(mpt1.r(I));     % maximal common bond size will be fixed                         
%                for i=1:I(I2)-2
%                    cores{i}=mpt1.cores{i}+mpt2.cores{i};
%                end
%                if I(I2)==1
%                    cores{mpt1.N}=mpt1.cores{mpt1.N}.rplus(mpt2.cores{mpt1.N});
%                    for i=I(I2)+1:mpt1.N-1
%                        cores{i}=mpt1.cores{i}+mpt2.cores{i};
%                    end
%                else
%                    cores{I(I2)-1}=mpt1.cores{I(I2)-1}.rplus(mpt2.cores{I(I2)-1});
%                    for i=I(I2)+1:mpt1.N
%                        cores{i}=mpt1.cores{i}+mpt2.cores{i};
%                    end
%                end
%                cores{I(I2)}=mpt1.cores{I(I2)}.lplus(mpt2.cores{I(I2)});
%            else
%                % both mpts have no common link dimensions, we therefore do
%                % not fix an "edge"
%                for i=1:mpt1.N
%                    cores{i}=mpt1.cores{i}+mpt2.cores{i};
%                end
%            end
%            mpt=MPT(cores);
        end
        
        function X=fwht(mps)
            % computes fast Walsh-Hadamard transform on vector in MPS form.
            % Hadamard ordering is used.
            if sum(mps.k-ones(1,length(mps.k)))~=0
                error('This method only works on an MPS.');
            end
            cores=cell(1,length(mps.k));
            for i=1:length(mps.k)
                cores{i}=Core([mps.r(i),mps.r(i+1)],mps.n{i},permute(reshape(fwht(mps.cores{i}.bottom2top,mps.n{i},'Hadamard'),[mps.n{i},mps.r(i+1),mps.r(i)]),[3,1,2]));
            end
            X=MPT(cores);            
        end

        %%   
        % -------------        
        % |   MISC    |
        % -------------    
              
        function mpt2=copy(mpt)
           % make a copy of a given MPT
           mpt2=MPT(mpt.cores,mpt.normcore);
        end
        
        function mps(mpt)
            % converts the mpt into an mps of the same number of cores by
            % grouping all indices together into 1 index
            for i=1:mpt.N
               mpt.setCore(i,Core([mpt.r(i),mpt.r(i+1)],prod(mpt.n{i}),mpt.cores{i}.rnr));
            end
        end
        
%         function mpsinv=pinvmat(mps,k,tol)
%            % computes the pseudoinverse of a matricization of an mps as described in tensor-based dynamic mode decomposition by klus, gelb, peitz and schutte.
%            % Matricization turns indices n{1} to n{k} as row index and n{k+1} up to n{d} as column index. 
%            if sum(mps.k-ones(1,length(mps.k)))~=0
%                error('This method only works on an MPS.');
%            end
%            mpsnorm=mps.copy;
%            mpsnorm.skmc(k);
%            [U,S,V]=svd(mpsnorm.cores{k}.right2left);
%            s = diag(S);
%            r = sum(s > tol)+1;
%            S=S(1:r,1:r);
%            mpsnorm.setCore(k,Core([mps.cores{k}.r(1),r],mps.cores{k}.n,reshape(U(:,1:r),[mps.cores{k}.r(1),mps.cores{k}.n,r])));
%            mpsnorm.setCore(k+1,Core([r,mps.cores{k+1}.r(2)],mps.cores{k+1}.n,  reshape(V(:,1:r)'*mpsnorm.cores{k+1}.left2right,[r,mps.cores{k+1}.n,mps.cores{k+1}.r(2)]))); 
%            mp
%         end
        
        function skmc(mpt,k)
           % puts a given mpt into site-k-mixed-canonical form
           if mpt.normcore==1
               % can move from site 1 to site k
               % orthogonalizes the mpt from left to right and puts the whole
               % norm of the tensor into the last core
               for i=1:k-1
                   [Q,R]=qr(mpt.cores{i}.right2left,0);
                   % make current core right-orthogonal
                   mpt.setCore(i,Core([mpt.r(i),numel(Q)/(mpt.r(i)*prod(mpt.n{i}))],mpt.n{i},reshape(Q,[mpt.r(i),mpt.n{i},numel(Q)/(mpt.r(i)*prod(mpt.n{i}))])));                
                   % propagate the norm to the next core
                   mpt.setCore(i+1,Core([numel(Q)/(mpt.r(i)*prod(mpt.n{i})),mpt.r(i+2)],mpt.n{i+1},reshape(R*mpt.cores{i+1}.left2right,[numel(Q)/(mpt.r(i)*prod(mpt.n{i})),mpt.n{i+1},mpt.r(i+2)])));                              
               end
               mpt.normcore=k;           
           elseif mpt.normcore==mpt.N
               % can move from site N to site k
               % orthogonalizes the mpt from right to left and puts the whole
               % norm of the tensor into the first core
               for i=mpt.N:-1:k+1
                   [Q,R]=qr(mpt.cores{i}.left2right',0);
                   % make current core right-orthogonal
                   mpt.setCore(i,Core([numel(Q)/(mpt.r(i+1)*prod(mpt.n{i})),mpt.r(i+1)],mpt.n{i},permute(reshape(Q,[mpt.n{i},mpt.r(i+1),numel(Q)/(mpt.r(i+1)*prod(mpt.n{i}))]),[mpt.k(i)+2,1:mpt.k(i),mpt.k(i)+1])));                
                   % propagate the norm to previous core
                   mpt.setCore(i-1,Core([mpt.r(i-1),numel(Q)/(mpt.r(i+1)*prod(mpt.n{i}))],mpt.n{i-1},reshape(mpt.cores{i-1}.right2left*R',[mpt.r(i-1),mpt.n{i-1},numel(Q)/(mpt.r(i+1)*prod(mpt.n{i}))])));                              
               end
               mpt.normcore=k;
           else
               mpt.rlortho;
               for i=1:k-1
                   [Q,R]=qr(mpt.cores{i}.right2left,0);
                   % make current core right-orthogonal
                   mpt.setCore(i,Core([mpt.r(i),numel(Q)/(mpt.r(i)*prod(mpt.n{i}))],mpt.n{i},reshape(Q,[mpt.r(i),mpt.n{i},numel(Q)/(mpt.r(i)*prod(mpt.n{i}))])));                
                   % propagate the norm to the next core
                   mpt.setCore(i+1,Core([numel(Q)/(mpt.r(i)*prod(mpt.n{i})),mpt.r(i+2)],mpt.n{i+1},reshape(R*mpt.cores{i+1}.left2right,[numel(Q)/(mpt.r(i)*prod(mpt.n{i})),mpt.n{i+1},mpt.r(i+2)])));                              
               end
               mpt.normcore=k;               
           end
        end
        
        function lrortho(mpt)
            % orthogonalizes the mpt from left to right and puts the whole
            % norm of the tensor into the last core
            for i=1:mpt.N-1
                [Q,R]=qr(mpt.cores{i}.right2left,0);
                % make current core right-orthogonal
                mpt.setCore(i,Core([mpt.r(i),numel(Q)/(mpt.r(i)*prod(mpt.n{i}))],mpt.n{i},reshape(Q,[mpt.r(i),mpt.n{i},numel(Q)/(mpt.r(i)*prod(mpt.n{i}))])));                
                % propagate the norm to the next core
                mpt.setCore(i+1,Core([numel(Q)/(mpt.r(i)*prod(mpt.n{i})),mpt.r(i+2)],mpt.n{i+1},reshape(R*mpt.cores{i+1}.left2right,[numel(Q)/(mpt.r(i)*prod(mpt.n{i})),mpt.n{i+1},mpt.r(i+2)])));                              
            end
            mpt.normcore=mpt.N;
        end        
        
        function rlortho(mpt)
            % orthogonalizes the mpt from right to left and puts the whole
            % norm of the tensor into the first core
            for i=mpt.N:-1:2
                [Q,R]=qr(mpt.cores{i}.left2right',0);
                % make current core right-orthogonal
                mpt.setCore(i,Core([numel(Q)/(mpt.r(i+1)*prod(mpt.n{i})),mpt.r(i+1)],mpt.n{i},permute(reshape(Q,[mpt.n{i},mpt.r(i+1),numel(Q)/(mpt.r(i+1)*prod(mpt.n{i}))]),[mpt.k(i)+2,1:mpt.k(i),mpt.k(i)+1])));                
                % propagate the norm to previous core
                mpt.setCore(i-1,Core([mpt.r(i-1),numel(Q)/(mpt.r(i+1)*prod(mpt.n{i}))],mpt.n{i-1},reshape(mpt.cores{i-1}.right2left*R',[mpt.r(i-1),mpt.n{i-1},numel(Q)/(mpt.r(i+1)*prod(mpt.n{i}))])));                              
            end
            mpt.normcore=1;
        end
        
        function round(mpt,varargin)
            % mpt.round or mpt.round(tol)
            % Rounds the MPT 
            
            % check whether mpt is in mixed-canonical form
            if mpt.normcore==0
                % it's not, so we orthogonalize and put the norm in the
                % last core
                mpt.lrortho;
            end
            
            if ~isempty(varargin)
                delta=mpt.norm*varargin{1}/sqrt(mpt.N*mpt.r(1));
            else
                % Default tolerance: ||mpt||_F * 1e-10/ sqrt(N-1)
                delta=mpt.norm*1e-10/sqrt(mpt.N-1);
            end            
            if mpt.normcore == mpt.N
                % SVD sweep from right-to-left
                for i=[mpt.N:-1:2]                
                    [U,S,V]=svd(mpt.cores{i}.left2right','econ');
                    s=diag(S);
                    % choose rk of SVD USV'+E such that ||E||_F <= delta
                    ri=1;
                    normE=norm(s(ri+1:end));
                    while normE > delta
                        ri=ri+1;
                        normE=norm(s(ri+1:end));
                    end
                    mpt.setCore(i,Core([ri,mpt.r(i+1)],mpt.n{i},reshape(U(:,1:ri)',[ri,mpt.n{i},mpt.r(i+1)])));
                    if i==1
                        % absorb S*V' into 1st core
                        mpt.setCore(mpt.N,Core([mpt.r(mpt.N),ri],mpt.n{mpt.N},reshape(mpt.cores{mpt.N}.right2left*V(:,1:ri)*S(1:ri,1:ri),[mpt.r(mpt.N),mpt.n{mpt.N},ri])));
                    else
                        mpt.setCore(i-1,Core([mpt.r(i-1),ri],mpt.n{i-1},reshape(mpt.cores{i-1}.right2left*V(:,1:ri)*S(1:ri,1:ri),[mpt.r(i-1),mpt.n{i-1},ri])));
                    end
                end
                mpt.normcore=1;
            elseif mpt.normcore==1
                % SVD sweep from left-to-right                
                for i=1:mpt.N-1               
                    [U,S,V]=svd(mpt.cores{i}.right2left,'econ');
                    s=diag(S);
                    % choose rk of SVD USV'+E such that ||E||_F <= delta
                    ri=1;
                    normE=norm(s(ri+1:end));
                    while normE > delta
                        ri=ri+1;
                        normE=norm(s(ri+1:end));
                    end
                    % update the ranks and core
                    mpt.setCore(i,Core([mpt.r(i),ri],mpt.n{i},reshape(U(:,1:ri),[mpt.r(i),mpt.n{i},ri])));
                    if i==mpt.N
                        %propagate S*V' to last core
                        mpt.setCore(1,Core([ri,mpt.r(2)],mpt.n{1},reshape(S(1:ri,1:ri)*V(:,1:ri)'*mpt.cores{1}.left2right,[ri,mpt.n{1},mpt.r(2)])));
                    else
                        mpt.setCore(i+1,Core([ri,mpt.r(i+2)],mpt.n{i+1},reshape(S(1:ri,1:ri)*V(:,1:ri)'*mpt.cores{i+1}.left2right,[ri,mpt.n{i+1},mpt.r(i+2)])));
                    end
                end
                mpt.normcore=mpt.N;
            end            
        end
        
        function [mpt2,e]=als_approx(mpt)
            % [mpt2,e]=als_approx(mpt,r) or [mpt2,e]=als_approx(mpt,mpt2)
            % uni-directional ALS. Can provided the vector of ranks r or an
            % initial guess mps2.
            [mpt2,e]=mptalsapprox(mpt);
        end
        
        function [mpt2,e]=mals_approx(mpt,k)
            [mpt2,e]=mptmalsapprox(mpt,k);
        end  
        
        function mpt2=fliplr(mpt)
           % flips the MPT left to right, so core_1--core_2--...--core_N
           % becomes core_N--...--core_2--core_1
           cores=cell(1,mpt.N);
           for i=1:mpt.N
               cores{i}=fliplr(mpt.cores{end-i+1});
           end
           mpt2=MPT(cores,mpt.normcore);
        end
        
        function setCore(mpt,k,core,varargin)
            % Changes the k-th core in the MPT, also changes the r vector
            % accordingly
            mpt.r(k)=core.r(1);
            mpt.r(k+1)=core.r(2);
            mpt.k(k)=core.k;
            mpt.n{k}=core.n;
            mpt.cores{k}=core; 
            % Need to check whether new core is also either left or
            % right-orthogonal and adjust normcore variable accordingly            
            if ~isempty(varargin)
                mpt.normcore=k;
            end
        end    
        
        function A=lprop(mpt,k,A)
           % Propagation from left-to-right of matrix A starting at core k
           % around the whole MPT.
           indices=[k:mpt.N 1:k-1];
           for i=indices
               [Q,A]=qr(reshape(A*mpt.cores{i}.left2right,[mpt.r(i)*mpt.n{i},mpt.r(i+1)]));
               A=A(1:mpt.r(i+1),:);
               % replace current core
               mpt.setCore(i,Core([mpt.r(i),mpt.r(i+1)],mpt.n{i},reshape(Q(:,1:mpt.r(i+1)),[mpt.r(i),mpt.n{i},mpt.r(i+1)])));
           end
           mpt.normcore=k;
        end
        
        function A=rprop(mpt,k,A)
           % Propagation from right-to-left of matrix A starting at core k
           % around the whole MPT.
           % make a local copy of mpt as the cores will change in this
           % procedure
           indices=[k-1:-1:1 mpt.N:-1:k];
           for i=indices
               [Q,A]=qr(reshape(mpt.cores{i}.right2left*A,[mpt.r(i),mpt.n{i}*mpt.r(i+1)])');
               A=A(1:mpt.r(i),:)';
               % replace current core
               mpt.setCore(i,Core([mpt.r(i),mpt.r(i+1)],mpt.n{i},reshape(Q(:,1:mpt.r(i))',[mpt.r(i),mpt.n{i},mpt.r(i+1)])));
           end
           mpt.normcore=k-1;
        end 
        
        function s=numel(mpt)
            % Returns the number of parameters that are required to store
            % the cores of the whole mpt in memory.
            s=sum(cellfun(@(x)numel(x.core),mpt.cores));
        end
        
        function [mera,e]=openMERA(mps,k,varargin)
            % [mera,e] = openMERA(mps,k) or [mera,e]=openMERA(mps,k,r)
            % --------------------------------------------------------
            % Initializes a MERA with open boundary conditions from a given mps through consecutive SVD
            % operations. This MERA can be used as an initial guess in the
            % optimization algorithm.
            % The k vector contains the number of incoming legs in the
            % isometries for each layer in the MERA.            
            % r vector or cell: The ith element of the r cell contains the outgoing dimensions of the
            % isometries in the ith layer of the MERA. In case r is a
            % vector then all isometries in layer i have identical output
            % r(i).
            % e is the relative approximation error of the MERA.
            if ~isempty(varargin)
                [mera,e]=opMERA(mps,k,varargin{1});
                e=e/mps.norm;
            else
                [mera,e]=opMERA(mps,k);
                e=e/mps.norm;
            end
        end
        
       function [Smps,U]=HOSVD(mps)
            % HOSVD, 
            % returns square orthogonal factor matrices and all-orthogonal Tucker
            % core in mps form.
           if sum(mps.k)~=mps.N
               error('This method only works for an MPS.');
           end
           if mps.normcore==0 || (mps.normcore ~= 1 && mps.normcore~=mps.N)
               warning('Mps needs to be in either site-1 or site-N mixed-canonical form, orthogonalizing now.');
               mps.rlortho;
           end
           U=cell(1,mps.N);         % this cell contains the Tucker factor matrices
           Scores=cell(1,mps.N);    % this cell contains the Core objects for the Tucker core MPS
           mps2=mps.copy;
           if mps.normcore==1
               % sweep from left-to-right
               for i=1:mps.N
                   [U{i},S,V]=svd(mps2.cores{i}.bottom2top);
                   temp=permute(reshape(S*V',[mps.n{i},mps.r(i+1),mps.r(i)]),[3,1,2]);
                   if i==mps.N
                       % retain the SV' factor as the last core in MPS of
                       % Tucker core
                       Scores{i}=Core([mps.r(i),mps.r(i+1)],mps.n{i},temp);
                   else
                       % Setup MPS-core i of Tucker-core
                       [Q,R]=qr(reshape(temp,[mps.r(i)*mps.n{i},mps.r(i+1)]),0);                       
                       Scores{i}=Core([mps.r(i),mps.r(i+1)],mps.n{i},reshape(Q,[mps.r(i),mps.n{i},mps.r(i+1)]));
                       % propagate the norm of the mps to the next MPS-core
                       mps2.setCore(i+1,Core([mps.r(i+1),mps.r(i+2)],mps.n{i+1},reshape(R*mps.cores{i+1}.left2right,[mps.r(i+1),mps.n{i+1},mps.r(i+2)])));
                   end                   
               end
               Smps=MPT(Scores,mps.N);
           elseif mps.normcore==mps.N
               % sweep from right-to-left
               for i=mps.N:-1:1
                   [W,S,U{i}]=svd(mps2.cores{i}.top2bottom);
                   temp=permute(reshape(W*S,[mps.r(i),mps.r(i+1),mps.n{i}]),[1,3,2]);
                   if i==1
                       % retain the W*S factor as the first core in MPS of
                       % Tucker core
                       Scores{i}=Core([mps.r(i),mps.r(i+1)],mps.n{i},temp);
                   else
                       % Setup MPS-core i of Tucker-core
                       [Q,R]=qr(reshape(temp,[mps.r(i),mps.n{i}*mps.r(i+1)])',0);                       
                       Scores{i}=Core([mps.r(i),mps.r(i+1)],mps.n{i},reshape(Q',[mps.r(i),mps.n{i},mps.r(i+1)]));
                       % propagate the norm of the mps to the next MPS-core
                       mps2.setCore(i-1,Core([mps.r(i-1),mps.r(i)],mps.n{i-1},reshape(mps.cores{i-1}.right2left*R',[mps.r(i-1),mps.n{i-1},mps.r(i)])));
                   end                   
               end
               Smps=MPT(Scores,1);
           end            
        end
        
        function [Smps,U,e]=THOSVD(mps,varargin)
            % [Smps,U,e]=THOSVD(mps) or [Smps,U,e]=THOSVD(mps,ranks) or [Smps,U,e]=THOSVD(mps,epsilon)
            % ----------------------------------------------------------------------------------------
            % Truncated-HOSVD, 
            % returns columnwise orthonormal factor matrices U{i} and all-orthogonal Tucker
            % core in mps form. Optional (mps.N x 1)-ranks vector can be provided to
            % indicate the desired dimensions of the factor matrices.
            % Providing ranks yourself may decrease the mps ranks further.
            % Other optional argument is an upperbound epsilon for the
            % relative approximation error.
            % e is the absolute approximation error.
           if sum(mps.k)~=mps.N
               error('This method only works for an MPS.');
           end
           if mps.normcore==0 || (mps.normcore ~= 1 && mps.normcore~=mps.N)
%                warning('Mps needs to be in either site-1 or site-N mixed-canonical form, orthogonalizing now.');
               mps.rlortho;
           end
           U=cell(1,mps.N);         % this cell contains the Tucker factor matrices
           Scores=cell(1,mps.N);    % this cell contains the Core objects for the Tucker core MPS
           mps2=mps.copy;
           if ~isempty(varargin) && isscalar(varargin{1})
               mpsnorm=mps.norm;
               delta=varargin{1}*mpsnorm/sqrt(mps.N);   % delta threshold in case upper bound for relative error was given.
           end
           e=0;
           if mps.normcore==1
               % sweep from left-to-right
               for i=1:mps.N
                   [W,S,V]=svd(mps2.cores{i}.bottom2top,'econ');
                   s=diag(S);
                   if ~isempty(varargin)
                       if ~isscalar(varargin{1})
                           % multilinear rank was provided by the user                     
                           r=varargin{1}(i);
                       else
                          % relative error was provided
                          r=1;
                          normE=norm(s(r+1:end));
                          while normE > delta
                              r=r+1;
                              normE=norm(s(r+1:end));
                          end
                       end
                   else
                       % no rank argument was provided by the user                       
                       tol=max(size(mps.cores{i}.bottom2top))*eps(max(s));
                       r=sum(s>tol);
                   end
                   U{i}=W(:,1:r); 
                   e=e+sum(s(r+1:end).^2);
                   temp=permute(reshape(S(1:r,1:r)*V(:,1:r)',[r,mps2.r(i+1),mps2.r(i)]),[3,1,2]);
                   if i==mps.N
                       % retain the SV' factor as the last core in MPS of
                       % Tucker core
                       Scores{i}=Core([mps2.r(i),mps2.r(i+1)],r,temp);
                   else
                       % Setup MPS-core i of Tucker-core
                       [Q,R]=qr(reshape(temp,[mps2.r(i)*r,mps2.r(i+1)]),0);                       
                       Scores{i}=Core([mps2.r(i),numel(Q)/(mps2.r(i)*r)],r,reshape(Q,[mps2.r(i),r,numel(Q)/(mps2.r(i)*r)]));
                       % propagate the norm of the mps to the next MPS-core
                       mps2.setCore(i+1,Core([numel(Q)/(mps2.r(i)*r),mps2.r(i+2)],mps.n{i+1},reshape(R*mps.cores{i+1}.left2right,[numel(Q)/(mps2.r(i)*r),mps2.n{i+1},mps2.r(i+2)])));
                   end                   
               end
               Smps=MPT(Scores,mps.N);
               e=sqrt(e);
%                e=sqrt(e)/mps.norm;
           elseif mps.normcore==mps.N
               % sweep from right-to-left
               for i=mps.N:-1:1
                   [W,S,V]=svd(mps2.cores{i}.top2bottom,'econ');
                   s=diag(S);
                   if ~isempty(varargin)
                       if ~isscalar(varargin{1})
                           % multilinear rank was provided by the user                     
                           r=varargin{1}(i);
                       else
                          % relative error was provided
                          r=1;
                          normE=norm(s(r+1:end));
                          while normE > delta
                              r=r+1;
                              normE=norm(s(r+1:end));
                          end
                       end
                   else                                             
                       tol=max(size(mps.cores{i}.top2bottom))*eps(max(s));
                       r=sum(s>tol);
                   end
                   U{i}=V(:,1:r);
                   e=e+sum(s(r+1:end).^2);
                   temp=permute(reshape(W(:,1:r)*S(1:r,1:r),[mps2.r(i),mps2.r(i+1),r]),[1,3,2]);
                   if i==1
                       % retain the W*S factor as the first core in MPS of
                       % Tucker core
                       Scores{i}=Core([mps2.r(i),mps2.r(i+1)],r,temp);
                   else
                       % Setup MPS-core i of Tucker-core
                       [Q,R]=qr(reshape(temp,[mps2.r(i),r*mps2.r(i+1)])',0);                       
                       Scores{i}=Core([numel(Q)/(r*mps2.r(i+1)),mps2.r(i+1)],r,reshape(Q',[numel(Q)/(r*mps2.r(i+1)),r,mps2.r(i+1)]));
                       % propagate the norm of the mps to the next MPS-core
                       mps2.setCore(i-1,Core([mps2.r(i-1),numel(Q)/(r*mps2.r(i+1))],mps.n{i-1},reshape(mps.cores{i-1}.right2left*R',[mps2.r(i-1),mps.n{i-1},numel(Q)/(r*mps2.r(i+1))])));
                   end                   
               end
               Smps=MPT(Scores,1);
               e=sqrt(e)/mps.norm;
%                e=sqrt(e)/mps.norm;
           end            
        end
        
        function mps=diag(mpo)
           % mps=diag(mpo)
           % -------------
           % Returns the main diagonal as an MPS of a square matrix in MPO form.
           % Dimension factorization of rows and columns need to be
           % identical.
           if sum(mpo.k)~=mpo.N*2
               error('This method only works for a given MPO.')
           end
           cores=cell(1,mpo.N);
           % need to determine diagonal element indices
           if sum(cellfun(@(x)x(1)-x(2),mpo.n))
              error('Only MPO of square matrices with uniform dimension factorization is supported.')    
           end
           for i=1:mpo.N
               temp=mpo.cores{i}.top2bottom;
               cores{i}=Core([mpo.r(i),mpo.r(i+1)],mpo.n{i}(1),permute(reshape(temp(:,1:mpo.n{1}(1)+1:mpo.n{1}(1)^2),[mpo.r(i),mpo.r(i+1),mpo.n{i}(1)]),[1,3,2]));
           end
           mps=MPT(cores);
        end       

        
%         function mpt=kron(mpt1,mpt2)
%            % mpt=kron(mpt1,mpt2)
%            % ------------------
%            % Constructs the MPT where each core is the Kronecker product of
%            % corresponding cores of mpt1 with mpt2
%            % do we want the order of the MPT to remain? (I'd think so)
%            cores=cell(1,mpt1.N);
%            for i=1:mpt1.N
%                cores{i}=mpt1.cores{i}.kron(mpt2.cores{i});
% %                temp=mpt1.cores{i}.kron(mpt2.cores{i});
% %                n=[mpt1.n{i},mpt2.n{i}];
% %                n=n([3,1,4,2]);
% %                cores{i}=Core(temp.r,n,reshape(temp.core,[temp.r(1),n,temp.r(2)]));
% %                temp=mpt1.cores{i}.kron(mpt2.cores{i}); 
%                
%                %% FIX ME, some permutation here needed
% %                cores{i}=Core(temp.r,mpt1.n{i}.*mpt2.n{i},reshape(temp.core,[temp.r(1),mpt1.n{i}.*mpt2.n{i},temp.r(2)]));
%            end
%            mpt=MPT(cores);
%         end

    end    
end