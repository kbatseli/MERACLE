classdef Isometry < handle
    % w=Isometry(k,nin,nout,tensor)
    % -----------------------------
    % Represents an isometry, a basic building block of a MERA. An isometry
    % is per definition a (k+1)-way tensor, where k is the number of bottom
    % legs, all of dimension nin. The top index has dimension nout.
    %
    % k         =   scalar, number of bottom legs,
    %
    % nin       =   scalar, dimension of each of the k bottom legs,
    %
    % nout      =   scalar, dimension of the top leg,
    %
    % tensor    =   (k+1)-way array.
    %
    % The ordering of the indices is
    %
    %       k+1
    %        |       top, 1 leg
    %       / \
    %      /   \
    %     /  w  \
    %    /       \
    %    ---------
    %    | | ... |  bottom, k legs
    %    1 2     k
    
    properties (SetAccess = private)
        k       % number of bottom legs
        nin     % vector, nin(i) dimension of bottom leg i
        nout    % dimension of outgoing leg
        core    % (k+1)-way tensor, indices ordered as indicated in diagram above
    end
    
    methods
        % constructor
        function w=Isometry(k,nin,nout,varargin)
            w.k=k;
            w.nin=nin;
            w.nout=nout;
            if ~isempty(varargin)
                w.setCore(varargin{1});
            end
        end
        
        function setCore(w,core)
            if w.nout==1
                score=[size(core),1];
            else
                score=size(core);
            end
            if length(score) ~= w.k+1 || sum(abs(score-[w.nin,w.nout])) > 0
                error(['Provided data should be of size ' num2str([w.nin,w.nout]) '.']);
            else
                w.core=core;
            end
        end

        %%   
        % ----------------        
        % |  RESHAPINGS  |
        % ----------------
        
        function t2b=top2bottom(w)
           % reshapes the isometry into a matrix where first mode is the top leg
           %
           % (k+1)---O---[1,2,...,k]
           %
           t2b=reshape(w.core,[prod(w.nin),w.nout])';
        end
        
        function b2t=bottom2top(w)
            % reshapes the isometry into a matrix where first mode are
            % all bottom indices grouped together
            %
            % [1,2,...,k]---O---(k+1)
            %            
            b2t=reshape(w.core,[prod(w.nin),w.nout]);
        end        
        
        %%   
        % ------------------        
        % |  CONTRACTIONS  |
        % ------------------    
        
        function core=bottomcon(w,core)
           % Contract the isometry over its bottom indices with a core.
           % Produces another Core with a single free index
           %
           %               n
           %               |
           %              / \
           %             /   \
           %            /  w  \
           %           /       \
           %           ---------
           %            | ... |  
           %            -------
           %      r(1)  |     |  r(2)
           %      ------|core |------- 
           %            |     |    
           %            -------    
           
           core=Core(core.r,w.nout,permute(reshape(w.top2bottom*core.bottom2top,[w.nout,fliplr(core.r)]),[3 1 2]));
        end
        
        function core=topcon(w,core)
           % Contract the isometry over its top index with a core.
           % Produces another Core with k free indices
           %
           %            -------
           %      r(1)  |     |  r(2)
           %      ------|core |------- 
           %            |     |    
           %            -------     
           %               |
           %               |
           %              / \
           %             /   \
           %            /  w  \
           %           /       \
           %           ---------
           %            | ... | 
           %           n(1)  n(k)
          
           core=Core(core.r,w.nin,permute(reshape(core.top2bottom*w.top2bottom,[core.r,w.nin]),[1,3:2+w.k,2]));
        end             
        
    end
    
end