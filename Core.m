classdef Core < handle
    % lcore=Core(r,n,tensor)
    % -----------------------
    % Represents a Tensor Network core with two auxiliary indices. If each core in
    % the chain has 1 free index, then the chain represents an MPS (matrix
    % product state), 2 free indices an MPO (matrix product operator),
    % etc... This generalizes to a MPT (matrix product tensor), where each
    % core has k free indices, resulting in a k-way tensor.
    %
    %   r       =   1x2 vector, r(1)= left-rank, r(2)=right-rank,
    %   n       =   n(k)= dimension of kth bottom leg,
    %   core    =   (k+2)-way tensor, indices ordered as indicated in
    %
    %           -------
    %      r(1) |     |  r(2)
    %   1 ------|     |------- k+2
    %           |     |    
    %           -------
    %           | ... |     bottom, k legs
    %           2    k+1   
    
    properties (SetAccess = private)
        r       % r(1)= left-rank, r(2)=right-rank
        n       % n(k)= dimension of kth bottom leg
        k       % number of bottom legs
        core    % (k+2)-way tensor, indices ordered as indicated in diagram above
    end
    
    methods
        % constructor
        function lcore=Core(varargin)
            % core=Core(A) or core=Core(r,n) or core=lcore(r,n,A)
            switch nargin
                case 1
                    if isa(varargin{1},'double')
                        % constructor core=Core(A)
                        lcore.r=[1 1];
                        lcore.n=size(varargin{1});
                        lcore.k=length(lcore.n);
                        lcore.core=varargin{1};
                    else
                        error('Single argument constructor of Core requires an argument of type ''double''.');
                    end
                otherwise
                    if length(varargin{1}) ~=2
                        error(['The input r should be a vector of length 2.'])
                    else
                        lcore.r=varargin{1};
                    end
                    if ~isempty(varargin{2})
                        lcore.n=varargin{2};
                        lcore.k=length(varargin{2});
                    else
                        lcore.n=1;
                        lcore.k=1;
                    end
                    if nargin==3
                        setCore(lcore,reshape(varargin{3},[lcore.r(1),lcore.n,lcore.r(2)]));
                    end
            end
        end        
        
        function setCore(lcore,core)
            lcore.core=core;
            % USER BEWARE!! We do not check whether the tensor your provide
            % has the right dimensions as indicated by the object
        end
        

        
        %%   
        % ----------------        
        % |  RESHAPINGS  |
        % ----------------
        
        function t2b=top2bottom(lcore)
           % reshapes the core into a matrix where first mode are 
           % the links grouped together as [r(1)r(2)] and the second mode 
           % consists of the free indices grouped together
           %
           %  [r(1),r(2)]---O---[n]
           %           
           t2b=reshape(permute(lcore.core,[1 lcore.k+2 2:lcore.k+1]),[prod(lcore.r),prod(lcore.n)]);
        end
        
        function b2t=bottom2top(lcore)
           % reshapes the core into a matrix where first mode are 
           % the free indices grouped together and the second mode consists
           % of the links grouped together as [r(2),r(1)]
           %
           %  [n]---O---[r(2),r(1)]
           %
           b2t=reshape(permute(lcore.core,[2:lcore.k+2 1]),[prod(lcore.n),prod(lcore.r)]);
        end
        
        function l2r=left2right(lcore)
            % reshapes the core into a matrix where the first mode
            % is the left linking index and the second mode are all
            % remaining indices grouped together: r(1) x [n,r(2)]
            %  
            % r(1)---O---[n,r(2)]
            %             
            l2r=reshape(lcore.core,[lcore.r(1),lcore.r(2)*prod(lcore.n)]);
        end
        
        function r2l=right2left(lcore)
            % reshapes the core into a matrix where the first mode
            % contains the left link and all free indices grouped together and
            % the second mode is the right link: [r(1),n] x r(2) 
            %  
            % [r(1),n]---O---r(2)
            % 
            r2l=reshape(lcore.core,[lcore.r(1)*prod(lcore.n),lcore.r(2)]);
        end
        
        function f2=first2(lcore)
            % reshapes the core into a matrix where the first mode are r(1)
            % and the first free index, and the second mode contains all
            % remaining free indices together with r(2)
            %  
            % [r(1),n(1)]---O---[n(2:end),r(2)]
            %             
            f2=reshape(lcore.core,[lcore.r(1)*lcore.n(1),prod(lcore.n(2:end))*lcore.r(2)]);            
        end        
        
        function fk=firstk(lcore,k)
            % reshapes the core into a matrix where the first mode are r(1)
            % and the first k-1 indices, and the second mode contains all
            % remaining free indices together with r(2). Do not confuse the scalar argument k
            % with property of the Core, 1 <= k <= lcore.k.
            %  
            % [r(1),n(1),..,n(k-1)]---O---[n(k:end),r(2)]
            %   
            fk=reshape(lcore.core,[lcore.r(1)*prod(lcore.n(1:k-1)),prod(lcore.n(k:end))*lcore.r(2)]);                        
        end
        
        function mk=modek(lcore,k)
            % reshapes the core into a matrix where the first mode are r(1)
            % and all free indices except the kth one, the second mode
            % corresponds with n(k), where k can be a vector containing multiple indices
            % Do not confuse the scalar argument k with the k-property of the Core,
            % 1 <= k <= lcore.k.
            %  
            % [r(1),n(1),..,n(k-1),n(k+1),...,n(end),r(2)]---O---n(k)
            %   
            nI=1:lcore.k; % indices for dimensions n
            nI(k)=[];
            I=1:lcore.k+2; % indices for permutation of whole core
            I(k+1)=[];
            mk=reshape(permute(lcore.core,[I,k+1]),[prod(lcore.r)*prod(lcore.n(nI)),prod(lcore.n(k))]);                        
        end
        
        function core=leftedge(topcore,bottomcore)
            % converts the following network of 2 Cores
            %
            % r(1) ---topcore---  r(2)
            %            |
            %            |            
            % s(1) --bottomcore-- s(2)
            %
            % into 1 Core
            %             __ r(2)
            %             |
            % r(1)*s(1)--core
            %             |
            %             -- s(2)
            %
            % such that core.r(1)= r(1)*s(1), core.r(2)=1 and core.n=[s(2),r(2)].
            core=topcore.topbottomcon(bottomcore);
            core.n=[bottomcore.r(2),topcore.r(2)];
            core.k=2;
            core.r(2)=1;
            core.core=reshape(permute(reshape(core.core,[topcore.r(1),bottomcore.r(1),topcore.r(2),bottomcore.r(2)]),[1 2 4 3]),[core.r(1),core.n,core.r(2)]);
        end
        
        function core=rightedge(topcore,bottomcore)
            % converts the following network of 2 Cores
            %
            % r(1) ---topcore---  r(2)
            %            |
            %            |            
            % s(1) --bottomcore-- s(2)
            %
            % into 1 Core
            %
            %    r(1) __
            %           |
            %     1---core-- r(2)*s(2)
            %           |
            %    s(1) --
            %
            % such that core.r(1)=1, core.n=[s(1),r(1)] and core.r(2)=r(2)*s(2).
            core=topcore.topbottomcon(bottomcore);
            core.r(1)=1;
            core.n=[bottomcore.r(1),topcore.r(1)];
            core.k=2;            
            core.core=reshape(permute(reshape(core.core,[topcore.r(1),bottomcore.r(1),topcore.r(2),bottomcore.r(2)]),[2 1 3 4]),[core.r(1),core.n,core.r(2)]);
        end   
        
        function rnr=rnr(core)
           % converts the core into a 3-way tensor of size r(1) x prod(n) x r(2) 
           rnr=reshape(core.core,[core.r(1),prod(core.n),core.r(2)]);
        end
        
        function cm=cyclicmode(core,i,m)
            % This method assumes that r(1)=r(2)=1 and reshapes the core
            % into a matrix. Before the reshaping, the first i-1 modes are
            % permuted in a cyclic manner so that the matrix has dimensions
            %  
            % [n(i),...,n(i+m-1)]---O---[n(i+m),..,n(d),n(1),...n(i-1)]
            if i<=core.k & m<core.k
                n=core.n;
                n=n([i:length(core.n),1:i-1]);
                cm=reshape(permute(reshape(core.core,core.n),[i:length(core.n),1:i-1]),[prod(n(1:m)),prod(n(m+1:end))]);
            else
                error('Number of permutations and number of indices to be combined need to be strictly smaller than the total number of indices of the Core.')
            end            
        end
        
        %%   
        % ------------------        
        % |  CONTRACTIONS  |
        % ------------------
        
        function core=leftncon(lcore,lcore2)
           % contracts lcore and lcore2 over their left link and all free
           % indices, that is, the summation goes over r(1) and all n(i)'s
           % of lcore and lcore2
           %        
           %                 [r(1),n]
           %  lcore.r(2)---O----------O----lcore2.r(2)
           %
           core=Core([lcore.r(2),lcore2.r(2)],1,reshape((lcore.right2left)'*(lcore2.right2left),[lcore.r(2),1,lcore2.r(2)]));
        end
        
        function core=topcon(lcore,lcore2)
           % contracts lcore with lcore2 over both links at the same time
           % in the order lcore-lcore2. Relevant for MPTs with periodic
           % boundary conditions, when the first and last core need to be
           % contracted over both links.
           %
           %        [r(2),r(1)]
           %  1---O-------------O----1
           %      |             |
           %      |             |
           %    lcore.n      lcore2.n
           %
           core=Core([1,1],[lcore.n lcore2.n],reshape(lcore.bottom2top*lcore2.top2bottom,[1 lcore.n lcore2.n 1]));
        end
                
        function core=leftcon(lcore,lcore2)
           % left contract
           % contracts lcore over its left link with lcore2
           %
           %           --------       --------
           %     r(1)  |      |       |      |  r(2)     
           %     ------|lcore2|-------|lcore |-------
           %           |      |       |      |
           %           --------       --------
           %           | ... |        | ... |
           %             
            core=Core([lcore2.r(1),lcore.r(2)],[lcore2.n lcore.n],reshape(lcore2.right2left*lcore.left2right,[lcore2.r(1),lcore2.n,lcore.n,lcore.r(2)]));
        end
        
        function core=rightcon(lcore,lcore2)
           % right contract
           % contracts lcore over its right link with lcore2
           %
           %           --------       --------
           %      r(1) |      |       |      |  r(2) 
           %     ------|lcore |-------|lcore2|-------
           %           |      |       |      |
           %           --------       --------
           %           | ... |        | ... |
           %     
            core=Core([lcore.r(1),lcore2.r(2)],[lcore.n lcore2.n],reshape(lcore.right2left*lcore2.left2right,[lcore.r(1),lcore.n,lcore2.n,lcore2.r(2)]));
        end   
        
        function core=topbottomcon(topcore,bottomcore)
            % converts the following network of Cores
            %
            %   r(1) ---topcore--- r(2)  
            %              |
            %              |   
            %  s(1)---bottomcore---s(2)
            %
            % into 1 Core
            %   
            % r(1)s(1)---O---r(2)s(2)
            %   
            % such that new r(1) is product of previous r(1)s and new r(2)
            % is product of previous r(2)s, results in either a scalar, vector or matrix.            
           core=Core(topcore.r.*bottomcore.r,1,reshape(permute(reshape(topcore.top2bottom*bottomcore.bottom2top,[topcore.r,fliplr(bottomcore.r)]),[1 4 2 3]),[topcore.r(1)*bottomcore.r(1),topcore.r(2)*bottomcore.r(2)]));
        end
        
        function core=modekcon(core1,k1,core2,k2)
           % contracts core1 over free mode k1 with core2 over its free k2 mode.
            %
            %           --------       
            %      r(1) |      |  r(2)     
            %     ------|core1 |-------
            %           |      |       
            %           --------       
            %           |..k1..|
            %              | 
            %              | 
            %           |..k2..|            
            %           --------       
            %      s(1) |      |  s(2)     
            %     ------|core2 |-------
            %           |      |       
            %           --------
            % 
            % into 1 Core
            %   
            %                --------       
            %      r(1)*s(1) |      |  r(2)*s(2)     
            %      ----------| core |------------
            %                |      |       
            %                --------       
            %                |..|..|
            %   
            % where the order of the free indices is
            % [core1.n(1:k1-1),core2.n(1:k2-1),core2.n(k2+1:end),core1.n(k1+1:end)],
            % which basically means that the remaining free indices of
            % core2 are inserted in between core1.n(k-1) and core1.n(k+1).
            temp=reshape(core1.modek(k1)*core2.modek(k2)',[core1.r(1),core1.n(1:k1-1),core1.n(k1+1:end),core1.r(2),core2.r(1),core2.n(1:k2-1),core2.n(k2+1:end),core2.r(2)]);
            temp=permute(temp,[1,core1.k+2,2:k1,core1.k+3:core1.k+core2.k+1,k1+1:core1.k+1,core1.k+core2.k+2]);
            core=Core([core1.r(1)*core2.r(1),core1.r(2)*core2.r(2)],[core1.n(1:k1-1),core2.n(1:k2-1),core2.n(k2+1:end),core1.n(k1+1:end)],reshape(temp,[core1.r(1)*core2.r(1),core1.n(1:k1-1),core2.n(1:k2-1),core2.n(k2+1:end),core1.n(k1+1:end),core1.r(2)*core2.r(2)]));
        end
                
        function tensor=contract(core)
           % contracts over the links and returns the resulting tensor 
           if core.r(1) ~= core.r(2)
               error('r(1) and r(2) need to be equal.')
           end
           tensor=core.top2bottom;
           tensor=reshape(sum(tensor(1:core.r(1)+1:core.r(1)^2,:),1),core.n);
        end        

        %%   
        % --------------        
        % |  SPLITS    |
        % --------------    
        
        function [mps,e]=mps(lcore,epsilon)
            % [mps,e]=mps(lcore,epsilon) or [mps,e]=mps(lcore,r)
            % --------------------------------------------------
            % splits an Core into an mps with relative approximation error
            % that is smaller than scalar argument epsilon.
            % r is a vector of length (lcore.k+2) that contains the desired
            % mps-ranks.
            % The output argument e is the obtained relative approximation error.
            d=lcore.k;
            cores=cell(1,d);
            frobnorm=norm(lcore.vec);
            if isscalar(epsilon)                
                delta=frobnorm*epsilon/sqrt(lcore.k-1);
                r=zeros(1,d+1);
                r(1)=lcore.r(1);
            else
                r=epsilon;
            end
            e=0;
            for i=1:d-1
                [U,S,V]=svd(lcore.first2,'econ');
                s=diag(S);
                if isscalar(epsilon)
                    % choose rk of SVD USV'+E such that ||E||_F <= delta
                    rk=1;
                    normE=norm(s(rk+1:end));
                    while normE > delta
                        rk=rk+1;
                        normE=norm(s(rk+1:end));
                    end
                    r(i+1)=rk;
                else
                   normE=norm(s(r(i+1)+1:end)); 
                end
                e=e+normE^2;                
                cores{i}=Core([r(i) r(i+1)],lcore.n(1),reshape(U(:,1:r(i+1)),[r(i),lcore.n(1),r(i+1)]));
                lcore=Core([r(i+1) lcore.r(2)],lcore.n(2:end),reshape(S(1:r(i+1),1:r(i+1))*V(:,1:r(i+1))',[r(i+1),lcore.n(2:end),lcore.r(2)]));                
            end
            cores{end}=lcore;
            mps=MPT(cores,d);
            e=sqrt(e)/frobnorm;
        end
        
        function mpt=ksplit(lcore,k,varargin)
            % mps=ksplit(lcore,k) or mps=ksplit(lcore,k,epsilon)
            % ----------------------------------------------
            % splits an Core with p*k free nodes into an mpt of p cores
            % where each core has k nodes. An optional relative approximation
            % error epsilon can be provided, default relative error is 1e-10.
            if ~isempty(varargin)
                delta=norm(lcore.vec)*varargin{1}/sqrt(lcore.k-1);
            else
                % default value for the tolerance
                delta=norm(lcore.vec)*1e-10/sqrt(lcore.k-1);
            end
            p=lcore.k/k;
            cores=cell(1,p);
            mptr=zeros(1,p);
            mptr(1)=lcore.r(1);
            for i=1:p-1
                [U,S,V]=svd(lcore.firstk(k+1),'econ');
                s=diag(S);
                % choose rk of SVD USV'+E such that ||E||_F <= delta
                rk=1;
                normE=norm(s(rk+1:end));
                while normE > delta
                    rk=rk+1;
                    normE=norm(s(rk+1:end));
                end
                mptr(i+1)=rk;
                cores{i}=Core([mptr(i),mptr(i+1)],lcore.n(1:k),reshape(U(:,1:mptr(i+1)),[mptr(i),lcore.n(1:k),mptr(i+1)]));
                lcore=Core([mptr(i+1),lcore.r(2)],lcore.n(k+1:end),reshape(S(1:mptr(i+1),1:mptr(i+1))*V(:,1:mptr(i+1))',[mptr(i+1),lcore.n(k+1:end),lcore.r(2)]));                                
            end
            cores{end}=lcore;
            mpt=MPT(cores,lcore.k);
        end
        
        function [mpt,e]=mptsvd(core,n,epsilon)
            % [mpt,e]=mptsvd(core,n,epsilon) or mpt=mptsvd(core,n,r)
            % ------------------------------------------------------
            % Converts a given k-way tensor into the MPT format using the 
            % mpt_svd algorithm.
            % n is a d x k matrix that specifies the dimensions of each
            % Core in the MPT and epsilon is the upperbound for the
            % relative approximation error.
            % r is a vector of length (d+2) that contains the
            % desired mpt-ranks.
            % e is the obtained 
            frobnorm=norm(core.vec);
            tensor=core.core;
            [d,k]=size(n);
            tensor=reshape(tensor,n(:)');
            indices=reshape([1:d*k],[d,k])';
            tensor=permute(tensor,indices(:));
            n2=prod(n,2)';
            tensor=reshape(tensor,n2);
            cores=cell(1,d);
            if isscalar(epsilon)
                % upperbound for the relative approximation error is given as input argument 
                delta=norm(core.vec)*epsilon/sqrt(d-1);
                mptr=zeros(1,d);
                mptr(1)=1;
            else
                % desired MPT-ranks are given as input argument
                mptr=epsilon;
            end            
            e=0;
            for i=1:d-1
                [U,S,V]=svd(reshape(tensor,[mptr(i)*n2(i),numel(tensor)/(mptr(i)*n2(i))]),'econ');
                s=diag(S);
                if isscalar(epsilon)
                    % choose rk of SVD USV'+E such that ||E||_F <= delta
                    rk=1;
                    normE=norm(s(rk+1:end));
                    while normE > delta
                        rk=rk+1;
                        normE=norm(s(rk+1:end));
                    end
                    mptr(i+1)=rk;
                else
                    normE=norm(s(mptr(i+1)+1:end));
                end
                e=e+normE^2;                
                cores{i}=Core([mptr(i),mptr(i+1)],n(i,:),reshape(U(:,1:mptr(i+1)),[mptr(i),n(i,:),mptr(i+1)]));
                tensor=reshape(S(1:mptr(i+1),1:mptr(i+1))*V(:,1:mptr(i+1))',[mptr(i+1),n2(i+1:end)]);
            end
            cores{d}=Core([mptr(i+1) 1],n(end,:),reshape(tensor,[mptr(i+1),n(end,:),1]));
            mpt=MPT(cores,d);
            e=sqrt(e)/frobnorm;
        end
        
        function mpt=rmptsvd(core,n,mptr,R)
            % mpt=rmptsvd(core,n,mptr,R)
            % --------------------------
            % Converts a given k-way tensor into the MPT format using the 
            % randomized mpt_svd algorithm.
            % n is a d x k matrix that specifies the dimensions of each
            % of the d Cores in the MPT.
            % mtpr is a 1x(d+1) vector of desired mpt-ranks with
            % mptr(1)=mptr(mpt.d+1)=1.
            % R is a vector of length (d-1) that contains the oversampling
            % parameters of mptr(2) up to mptr(d).
            tensor=core.core;
            [d,k]=size(n);
            tensor=reshape(tensor,n(:)');
            indices=reshape([1:d*k],[d,k])';
            tensor=permute(tensor,indices(:));
            n2=prod(n,2)';
            tensor=reshape(tensor,n2);
            cores=cell(1,d);
            for i=1:d-1
                [U,~]=qr(reshape(tensor,[mptr(i)*n2(i),numel(tensor)/(mptr(i)*n2(i))])*randn(numel(tensor)/(mptr(i)*n2(i)),mptr(i+1)+R(i)),0);
                U=U(:,1:mptr(i+1));
                cores{i}=Core([mptr(i),mptr(i+1)],n(i,:),reshape(U,[mptr(i),n(i,:),mptr(i+1)]));
                tensor=U'*reshape(tensor,[mptr(i)*n2(i),numel(tensor)/(mptr(i)*n2(i))]);
            end
            cores{d}=Core([mptr(i+1) 1],n(end,:),reshape(tensor,[mptr(i+1),n(end,:),1]));
            mpt=MPT(cores,d);
        end   
        
      
        
        function [topcore,bottomcore]=vsplit(core)
           % Vertical split.
           % Splits a 4-way Core with r(1)=r(2)=1 and ordering of free indices
           %    
           %    2  4
           %    |  |
           %    ----
           %   |core| 
           %    ----
           %    |  |
           %    1  3
           %
           % into 2 Cores
           %
           %  2---topcore--- 4
           %         |
           %         n
           %         |
           % 1--bottomcore---3 
           %
           temp=reshape(permute(reshape(core.core,core.n),[1 3 2 4]),[core.n(1)*core.n(3),core.n(2)*core.n(4)]);
           [U,S,V]=svd(temp,'econ');
           tol=S(1,1)*eps*max(size(temp));
           s=diag(S);
           r=sum(s>tol);
           bottomcore=Core([core.n(1),core.n(3)],r,permute(reshape(U(:,1:r),[core.n(1),core.n(3),r]),[1 3 2]));
           topcore=Core([core.n(2),core.n(4)],r,permute(reshape(S(1:r,1:r)*V(:,1:r)',[r,core.n(2),core.n(4)]),[2 1 3]));
        end
        
        %%   
        % ----------------------------        
        % |   OVERLOADED OPERATORS   |
        % ----------------------------    
        
        function core=plus(core1,core2)
           % adds two cores together, free indices need to have the same
           % dimensions
           if length(core1.n) ~= length(core2.n) || sum(abs(core1.n-core2.n))~=0
               error('Both cores need to have equal core.n.');
           end
           temp=core1.rnr;
           temp(core1.r(1)+1:core1.r(1)+core2.r(1),:,core1.r(2)+1:core1.r(2)+core2.r(2))=core2.rnr;           
           core=Core([core1.r(1)+core2.r(1),core1.r(2)+core2.r(2)],core1.n,reshape(temp,[core1.r(1)+core2.r(1),core1.n,core1.r(2)+core2.r(2)]));
        end   
        
        function core=uminus(core1)
           core=Core(core1.r,core1.n,-core1.core); 
        end
        
        function core=uplus(core1)
           core=Core(core1.r,core1.n,core1.core); 
        end    
        
        function core=minus(core1,core2)
           if isa(core1,'Core') && isa(core2,'Core')
              core=core1+(-core2); 
           else
                error('One of the provided arguments was not an Core.');
           end
        end        
        function core=mtimes(core1,a)
            % multiplication of mpt with scalar
            if (isa(core1,'Core') && isscalar(a))
                core=Core(core1.r,core1.n);
                core.setCore(a*core1.core);
            elseif (isa(a,'Core') && isscalar(core1))
                core=Core(a.r,a.n);
                core.setCore(core1*a.core);                
            end
        end

        function mpt=horzcat(varargin)
            % Turns [core_1 core_2 .... core_n] into corresponding MPT.
            % mpt=horzcat(core_1,core_2,...,core_n)            
            % concatenates core_1 with core_2 up to core_n to obtain an mpt            
            % all links should have matching dimensions
            varargin=varargin(~cellfun('isempty',varargin));    % remove empty entries in varargin
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
            % Turns [core_1 core_2 .... core_n] into corresponding MPT.            
            % mpt=vertcat(core_1,core_2,...,core_n)            
            % concatenates core_1 with core_2 up to core_n to obtain an mpt            
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
        %%   
        % -------------        
        % |   MISC    |
        % -------------            
        
        function v=vec(lcore)
            % vectorizes the data tensor 
            v=lcore.core(:);
        end
        
        function n=norm(lcore)
            % Frobenius norm of the tensor this core represents
           n=lcore.contract;
           n=norm(n(:));
        end
        
        function s=numel(core)
           s=numel(core.core); 
        end
        
        function core=lplus(core1,core2)
           % add two cores while keeping r(1) fixed over both cores.
           % Requires both cores to have equal r(1) and n.
           if length(core1.n) ~= length(core2.n) || sum(abs(core1.n-core2.n))~=0
               error('Both cores need to have equal core.n.');
           end
           if core1.r(1) ~= core2.r(1)
               error('Both cores need to have the same r(1).');
           end          
           core=Core([core1.r(1) core1.r(2)+core2.r(2)],core1.n,reshape([core1.right2left core2.right2left],[core1.r(1) core1.n core1.r(2)+core2.r(2)]));
        end
        
        function core=rplus(core1,core2)
           % adds two cores while keeping r(2) fixed over both cores. 
           % Requires both cores to have equal r(2) and n.  
           if length(core1.n) ~= length(core2.n) || sum(abs(core1.n-core2.n))~=0
               error('Both cores need to have equal core.n.');           end
           if core1.r(2) ~= core2.r(2)
               error('Both cores need to have the same r(2).');
           end
           core=Core([core1.r(1)+core2.r(1),core1.r(2)],core1.n,reshape([core1.left2right;core2.left2right] ,[core1.r(1)+core2.r(1),core1.n,core1.r(2)]));
        end       

        function core2=fliplr(core)
            % flips the orientation of the core r(1) x n x r(2) into
            % r(2) x fliplr(n) x r(1)
            core2=Core([core.r(2) core.r(1)],fliplr(core.n),permute(core.core,[core.k+2:-1:1]));
        end
        
        function squeeze(core)
           % removes singleton dimensions of the k bottom legs. If all
           % k bottom legs are singleton dimensions then k is reduced to 1.
           if sum(core.n)==core.k
               % all k legs are singletons, reduce to 1
               core.k=1;
               core.n=1;
               core.core=reshape(core.core,[core.r(1),1,core.r(2)]);               
           else               
               core.n=core.n(core.n>1);
               core.k=length(core.n);
               core.core=reshape(core.core,[core.r(1),core.n,core.r(2)]);
           end
        end
        
        function core=kron(core1,core2)
            % core=kron(core1,core2)
            % ----------------------
            % (right) Kronecker product of core1 with core2
            n1=[core1.r(1),core1.n,core1.r(2)];
            n2=[core2.r(1),core2.n,core2.r(2)];
            d1=length(n1);
            d2=length(n2);            
            % append with ones if the dimensions are not the same (due to Matlab's
            % removal of trailing ones in the dimension)
            if d1 > d2
                n2=[n2,ones(1,d1-d2)];
            elseif d2 > d1
                n1=[n1,ones(1,d2-d1)];
            end            
            c=kron(core1.vec,core2.vec); %compute all entries
            % now reshape the vector into the desired tensor
            permI=reshape(1:2*max(d1,d2),[max(d1,d2),2])';
            permI=permI(:)';
%             permI=[];
%             for i=1:max(d1,d2)
%                 permI=[permI i i+max(d1,d2)];
%             end
            core=Core([core1.r.*core2.r],core2.n.*core1.n,reshape(permute(reshape(c,[n2 n1]),permI),n1.*n2));
        end
        
        function [Score,factorm]=hosvd(core,varargin)
           % [Score,factorm]=hosvd(core) or [Score,factorm]=hosvd(core,core_size)
           % ------------------------------------------------------------------------
           % Multilinear svd of Core object, Score is an Core object of
           % the Tucker core.
           % factorm is a cell such that factorm{i} contains the factor
           % matrix of the ith mode as an Core. The concatenation
           % 
           % [ factorm{1} Score factorm{end}]
           %
           % will work and results in an MPT. The Cores of the factor
           % matrices corresponding with the free indices have the
           % following structure
           %
           %                --------       
           %          1     |      |      1     
           %      ----------|      |------------
           %                |      |       
           %                --------       
           %                |     |       
           %              core  Score
           %
           % where the first free index connects to the original core and the
           % second free index connects to the Score.
           % core_size is optional argument that specifies the desired dimensions
           % of the Tucker core. 
           if ~isempty(varargin) && length(varargin{1})==core.k+2
               core_size=varargin{1};
           else
               core_size=[core.r(1) core.n core.r(2)];
           end
           factorm=cell(1,core.k+2);
           A=core.core;
           Score=A;
           Adim=[core.r(1) core.n core.r(2)];
           Sdim=Adim;
           for i=1:core.k+2
               if core_size(i)==1
                   factorm{i}=Core([1 1],1,1);
               else
                  [U,~,~]=svd(reshape(A,[Adim(1),prod(Adim(2:end))]));
                  if i==1
                      % left link
                      factorm{i}=Core([Adim(1),core_size(i)],1,reshape(U(:,1:core_size(i)),[Adim(1),1,core_size(i)]));
                  elseif i==core.k+2
                      % right link
                      factorm{i}=Core([core_size(i),Adim(1)],1,reshape(U(:,1:core_size(i))',[core_size(i),1,Adim(1)]));
                  else
                      % free index
                      factorm{i}=Core([1,1],[Adim(1),core_size(i)],reshape(U(:,1:core_size(i)),[1 Adim(1),core_size(i),1]));    
                  end
                  Score=reshape(U(:,1:core_size(i))'*reshape(Score,[Adim(1),prod(Sdim(2:end))]),[core_size(i),Sdim(2:end)]);
                  Sdim(1)=core_size(i);
               end
               A=permute(A,[2:core.k+2 1]);
               Score=permute(Score,[2:core.k+2 1]);               
               Adim=Adim([2:core.k+2 1]);
               Sdim=Sdim([2:core.k+2 1]);
           end
           Score=Core([core_size(1),core_size(end)],core_size(2:end-1),Score);                      
        end
        
        function mptDim(lcore,n)
           % converts a Core into MPT dimension by doing required
           % permutation and reshaping. This assumes that lcore.n(1:d:end)
           % are the first mode dimensions, lcore.n(2:d:end) second mode,
           % etc...
           indices=reshape(2:lcore.k+1,[length(n),lcore.k/length(n)])';
           indices=indices(:)';
           lcore.setCore(reshape(permute(lcore.core,[1,indices,lcore.k+2]),[lcore.r(1),n,lcore.r(2)]));           
           lcore.n=n;
           lcore.k=length(n);
           
        end
        
        function [mps,varargout]=mpsals(core,mps)
            % Approximates this Core by an MPS via a 1/2-sweep ALS procedure,
            % starting from an initial MPT mps. This algorithm relies on the
            % site-1 or site-N-mixed-canonical form and therefore does not work when
            % there is a loop, this means that core.r(1)=core.r(2)=1 must be satisfied.
            if nargout==2
                ref=core.vec;         
                hat=mps.contract;
                varargout{1}(1)=norm(ref-hat)/norm(ref);
            end
            
            % construct a tensor from the core that has as many indices as the mps dictates
            n=cell2mat(mps.n);
            core=reshape(core.core,n);
            
            if mps.normcore==1   % left-to-right sweep
                
                % pre-compute all contractions with mps-cores to the right
                rcore=cell(1,mps.N+1);
                rcore{1,mps.N+1}=core;
                r_rnr=zeros(mps.N+1,mps.N+2);
                r_rnr(end,:)=[1,n,1];
                for j=mps.N:-1:2
                    rcore{j}=reshape(rcore{j+1},[prod(r_rnr(j+1,1:j)),prod(r_rnr(j+1,j+1:end))]);
                    rcore{j}=rcore{j}*mps.cores{j}.left2right';                                 %  n(1:j-1) x mps.r(j)
                    r_rnr(j,:)=r_rnr(j+1,:);
                    r_rnr(j,j+1:j+2)=[mps.r(j),1];
                end
                rcore{end}=[];
                
                for i=1:mps.N-1                                                              % index i refers to which mps-core gets updated
                    newcore=rcore{i+1};
                    l_rnr=r_rnr(i+1,:);
                    % contractions left of mps-core-i
                    for j=1:i-1
                        newcore=reshape(newcore,[prod(l_rnr(1:j+1)),prod(l_rnr(j+2:end))]);         % mps.r(j)*n(j) x n(j+1:end)                  
                        newcore=mps.cores{j}.right2left'*newcore;                                           % mps.r(j+1) x n(j+1:end)   
                        l_rnr(j:j+1)=[1,mps.r(j+1)];
                    end
                    
                    [Q,R]=qr(reshape(newcore,[mps.r(i)*mps.n{i},mps.r(i+1)]),0);     % produce left-orthogonal cores
                    mps.setCore(i,Core([mps.r(i),mps.r(i+1)],mps.n{i},reshape(Q,[mps.r(i),mps.n{i},mps.r(i+1)])));
                    mps.setCore(i+1,Core([mps.r(i+1),mps.r(i+2)],mps.n{i+1},reshape(R*mps.cores{i+1}.left2right,[mps.r(i+1),mps.n{i+1},mps.r(i+2)])),i+1);   % core i+1 contains the norm  
                    
                    if nargout==2    
                        hat=mps.contract;
                        varargout{1}(i+1)=norm(ref-hat)/norm(ref);
                    end
                end
            elseif mps.normcore==mps.N   % right-to-left sweep
                
                % pre-compute all contractions with mps-cores to the left
                lcore=cell(1,mps.N+1);
                lcore{1,1}=core;
                l_rnr=zeros(mps.N+1,mps.N+2);
                l_rnr(1,:)=[1,n,1];
                for j=1:mps.N-1
                    lcore{j+1}=reshape(lcore{j},[prod(l_rnr(j,1:j+1)),prod(l_rnr(j,j+2:end))]);       % mps.r(j)*n(j) x n(j+1:end)                  
                    lcore{j+1}=mps.cores{j}.right2left'*lcore{j+1};                                 % mps.r(j+1) x n(j+1:end)   
                    l_rnr(j+1,:)=l_rnr(j,:);
                    l_rnr(j+1,j:j+1)=[1,mps.r(j+1)];
                end
                
                for i=mps.N:-1:2 % contractions right of mps-core-i, index i refers to which mps-core gets updated
                    newcore=lcore{i};
                    r_rnr=l_rnr(i,:);
                    
                    for j=mps.N:-1:i+1                                                       
                        newcore=reshape(newcore,[prod(r_rnr(1:j)),prod(r_rnr(j+1:end))]);
                        newcore=newcore*mps.cores{j}.left2right';                            %  n(1:j-1) x mps.r(j)
                        r_rnr(j+1:j+2)=[mps.r(j),1];
                    end
                    [Q,R]=qr(reshape(newcore,[mps.r(i),mps.n{i}*mps.r(i+1)])',0);     % produce right-orthogonal cores
                    mps.setCore(i,Core([mps.r(i),mps.r(i+1)],mps.n{i},reshape(Q',[mps.r(i),mps.n{i},mps.r(i+1)])));
                    mps.setCore(i-1,Core([mps.r(i-1),mps.r(i)],mps.n{i-1},reshape(mps.cores{i-1}.right2left*R',[mps.r(i-1),mps.n{i-1},mps.r(i)])),i-1);   % core i+1 contains the norm  
                    
                    if nargout==2    
                        hat=mps.contract;
                        varargout{1}(i+1)=norm(ref-hat)/norm(ref);
                    end
                end
            end
        end
        
    end
    
end