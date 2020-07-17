classdef Disentangler < handle
    % u=Disentangler(n,tensor)
    % ------------------------
    % Represents a disentangler, a basic building block of a MERA. A
    % disentangler is per definition a 4-way tensor that can be reshaped
    % into a square orthogonal matrix.
    %
    % n         =   scalar, dimension of each of the 4 legs,
    %
    % tensor    =   4-way array.
    %
    % The ordering of the indices of the 4-way array is
    %
    %     3   4   
    %     |   |     top, 2 legs    
    %    -------
    %    |     |
    %    |  u  |
    %    |     |    
    %    -------
    %     |   |     bottom, 2 legs
    %     1   2   
    
    properties (SetAccess = private)
        n       % vector, n(i) is dimension of ith index, with n(1)=n(3) and n(2)=n(4)
        core    % 4-way tensor, indices ordered as indicated in diagram above
    end
    
    methods
        % constructor
        function u=Disentangler(n,varargin)
            if n(1)~=n(3) || n(2)~=n(4)
                error('First index dimension should be equal to third index dimension and second should be equal to the fourth.')
            else
                u.n=n;
            end
            if ~isempty(varargin)
                setCore(u,varargin{1});
            end
        end
        
        function setCore(u,core)
            if numel(core) ~= prod(u.n) || sum(abs(size(core)-u.n)) > 0 || u.n(1)~=u.n(3) || u.n(2)~=u.n(4)
                error(['Provided data should be of size ' num2str(u.n) '.']);
            else
                u.core=core;
            end
        end
        
        %%   
        % ----------------        
        % |  RESHAPINGS  |
        % ----------------
        
        function t2b=top2bottom(u)
           % reshapes the disentangler into a matrix where first mode are 
           % the top legs grouped together
           %
           % [3,4]---O----[1,2]
           %
           t2b=reshape(u.core,[prod(u.n(1:2)),prod(u.n(3:4))])';
        end
        
        function b2t=bottom2top(u)
            % reshapes the disentangler into a matrix where first mode are
            % the bottom indices grouped together
            %
            % [1,2]---O----[3,4]
            %            
            b2t=reshape(u.core,[prod(u.n(1:2)),prod(u.n(3:4))]);
        end
    
        
        %%   
        % ------------------        
        % |  CONTRACTIONS  |
        % ------------------        
        
        function core=bottomcon(u,core)
           % contract the disentangler over its bottom indices with a core
           % produces another Core
           %
           %           n(3) n(4)   
           %             |   |  
           %            -------
           %            |     |
           %            |  u  |
           %            |     |    
           %            -------
           %             |   |     
           %            -------
           %      r(1)  |     |    r(2)
           %      ------|core |------- 
           %            |     |    
           %            -------  
           %
           core=Core(core.r,u.n(3:4),permute(reshape(u.top2bottom*core.bottom2top,[u.n(3:4),fliplr(core.r)]),[4,1,2,3]));
        end
        
        function core=topcon(u,core)
           % contract the disentangler over its top indices with a Core
           % produces another Core
           %
           %            -------
           %      r(1)  |     |    r(2)
           %      ------|core |------- 
           %            |     |    
           %            -------    
           %             |   |  
           %            -------
           %            |     |
           %            |  u  |
           %            |     |    
           %            -------
           %             |   |     
           %           n(1) n(2)
           % 
           core=Core(core.r,u.n(1:2),permute(reshape(core.top2bottom*u.top2bottom,[core.r,u.n(1:2)]),[1,3,4,2]));
        end        
        
    end
    
end