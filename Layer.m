classdef Layer < handle
    % layer=Layer(U,W)
    % -------------------------
    % Represents a layer of a MERA, periodic or open. A periodic layer
    % is possible when the total number of bottom legs N is a multiple of the
    % incoming legs of the isometries. For example, when N=8 and k=2, then
    % its corresponding periodic layer is
    %
    %           |            |           |           |
    %           |            |           |           |
    %          / \          / \         / \         / \
    %         /   \        /   \       /   \       /   \
    %        /  w1 \      /  w2 \     /  w3 \     /  w4 \
    %       /       \    /       \   /       \   /       \
    %       ---------    ---------   ---------   ---------
    %  ___    |   |__     _|   |_     _|   |_     _|   |__    
    %     |   |      |   |       |   |       |   | 
    %     |   |      |   |       |   |       |   | 
    %    -------    -------     -------     -------
    %    |     |    |     |     |     |     |     |
    %    | u1  |    | u2  |     | u3  |     | u4  |
    %    |     |    |     |     |     |     |     |  
    %    -------    -------     -------     -------
    %     |   |      |   |       |   |       |   |
    %           
    %
    % A periodic layer always has an equal amount of disentanglers and
    % isometries, viz. N/k. The corresponding open layer for N=8, k=2 is
    %
    %           |            |           |           |
    %           |            |           |           |
    %          / \          / \         / \         / \
    %         /   \        /   \       /   \       /   \
    %        /  w1 \      /  w2 \     /  w3 \     /  w4 \
    %       /       \    /       \   /       \   /       \
    %       ---------    ---------   ---------   ---------
    %         |   |__     _|   |_     _|   |_     _|   |    
    %         |      |   |       |   |       |   |     |
    %         |      |   |       |   |       |   |     |
    %         |     -------     -------     -------    |
    %         |     |     |     |     |     |     |    |
    %         |     | u1  |     | u2  |     | u3  |    |
    %         |     |     |     |     |     |     |    |
    %         |     -------     -------     -------    |
    %         |      |   |       |   |       |   |     |
    %           
   
    properties (SetAccess = private)
        U           % U{i} contains the ith disentangler
        W           % W{i} contains the ith isometry
        N           % number of bottom legs
        k           % number of legs going into the isometries
        nin         % vector of length N, bottom dimensions
        nout        % vector of length N/k, top dimensions
    end
    
    methods
        % constructor
        function layer=Layer(U,W)
            % check whether all isometries have equal k bottom legs
            k=cellfun(@(x) x.k,W);
            if k(1)*length(k)~=sum(k)
                error('Isometries do not have equal number of bottom legs');
            end
            layer.k=k(1);
            layer.N=layer.k*length(W);            
            layer.U=U;
            layer.W=W;     
            layer.nin=cell2mat(cellfun(@(x) x.nin,W,'UniformOutput',0));
            layer.nout=cellfun(@(x) x.nout,W);
        end
        
        %%   
        % ------------------        
        % |  CONTRACTIONS  |
        % ------------------
        
        function mps2=bottomcon(layer,mps,epsilon)
            % contracts the MERA layer with an MPS from the bottom, returns
            % an MPS with N/k cores
            % epsilon is an upperbound on the relative approximation error
            % when splitting the tensors into separate TT-cores. 
            mps2=bconop(layer,mps,epsilon);            
        end
        
        function mps2=topcon(layer,mps,epsilon)
           % contracts the MERA layer with an MPS from the top returns an 
           % MPS  with N*k cores
           % epsilon is an upperbound on the relative approximation error
           % when splitting the tensors into separate TT-cores.           
           mps2=tconop(layer,mps,epsilon);           
        end
        
        function core=environminusu(layer,topmps,bottommps,u)
           % Computes the environment of disentangler u when the layer is
           % contracted with an mps both on the top and bottom
            if layer.periodic
                core=emuper(layer,topmps,bottommps,u); 
            else
                core=emu(layer,topmps,bottommps,u);
            end           
        end    
        
        function core=environminusw(layer,topmps,bottommps,w)
           % Computes the environment of isometry w when the layer is
           % contracted with an mps both on the top and bottom
            if layer.periodic
                core=emwper(layer,topmps,bottommps,w); 
            else
                core=emw(layer,topmps,bottommps,w);
            end           
        end
        
        
        %%   
        % -------------        
        % |   MISC    |
        % -------------         
        function optimizeU(layer,topmps,bottommps,u)
            core=layer.environminusu(topmps,bottommps,u);
            [U,S,V]=svd(reshape(core.core,[prod(core.n(1:2)),prod(core.n(3:4))]));
            layer.U{u}.setCore(reshape(U*V',core.n));
        end
        
        function optimizeW(layer,topmps,bottommps,w)
            core=layer.environminusw(topmps,bottommps,w);
            [U,S,V]=svd(reshape(core.core,[prod(core.n(1:layer.k)),core.n(end)]));     
            layer.W{w}.setCore(reshape(U(:,1:core.n(end))*V',core.n));
        end        
        
        function s=numel(layer)
           s=sum(cellfun(@(x)numel(x.core),layer.U))+sum(cellfun(@(x)numel(x.core),layer.W));
        end
        
    end
    
end