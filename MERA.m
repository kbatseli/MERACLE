classdef MERA < handle
    % mera=MERA(layers,top)
    % ---------------------
    % represent a MERA that consists of multiple layers and a top tensor
    
    properties (SetAccess = private)
        l           % number of layers
        layers      % cell of layers that build up the MERA        
        top         % top tensor, is a Core object
    end
    
    methods
        function mera=MERA(layers,top)
            % mera=MERA(layers,top)            
            % check whether layers match
            if ~iscell(layers)
                layers={layers};
            end
            mera.l=length(layers);            
            for i=1:mera.l-1
                if layers{i}.N/layers{i}.k ~= layers{i+1}.N
%                 outgoing=cellfun(@(x) x.nout,layers{i}.W);
%                 incoming=kron(cellfun(@(x) x.n,layers{i+1}.U),[1 1]);
%                 if (length(outgoing)~=length(incoming)) || sum(abs(outgoing-incoming))~=0
                    error(['Layer ' num2str(i) ' does not match with layer ' num2str(i+1) '.']);
                end
            end
            mera.layers=layers;
            % check whether top fits with top layer
            outgoing=cellfun(@(x) x.nout,mera.layers{end}.W);            
            if (length(outgoing)~=length(top.n)) || sum(abs(outgoing-top.n))~=0
                error(['Top core does not fit with top layer']);
            end
            mera.top=top;
        end
        
        function rele=optimize(mera,mps,hsweeps,vsweeps)
            % optimizes each core of the MERA to approximate a given mps
            % initialization stage,
            % 1 vsweep means from top to bottom and back to top
            % 1 hsweeps means from left to right
            
            % initialize each layer hsweeps times
            mps_norm=mps.norm;
%             rele=1;           
            rele(1)=norm(mps.contract-mera.contract)/mps_norm;
            BOTTOM=cell(1,mera.l+1);
            TOP=cell(1,mera.l+1);
            BOTTOM{1}=mps;
            for i=1:mera.l
                BOTTOM{1+i}=mera.layers{i}.bottomcon(BOTTOM{i});
                for j=1:hsweeps
                    for k=1:length(mera.layers{i}.W)
                        mera.layers{i}.optimizeW(BOTTOM{1+i},BOTTOM{i},k);
                        rele=[rele norm(mps.contract-mera.contract)/mps_norm];
                    end
                    for k=1:length(mera.layers{i}.U)
                        mera.layers{i}.optimizeU(BOTTOM{1+i},BOTTOM{i},k);
                        rele=[rele norm(mps.contract-mera.contract)/mps_norm];
                    end                    
                end  
                BOTTOM{1+i}=mera.layers{i}.bottomcon(BOTTOM{i});
            end
            % optimize the top tensor
%             mera.top=lCore([1 1],mera.top.n,reshape(BOTTOM{mera.l+1}.contract/BOTTOM{mera.l+1}.norm*mps_norm,mera.top.n));
            mera.top=lCore([1 1],mera.top.n,reshape(BOTTOM{mera.l+1}.contract,mera.top.n));
            rele=[rele norm(mps.contract-mera.contract)/mps_norm];
            TOP{end}=mera.top.split;
            
            % now converge each layer vsweeps times            
            for j=1:vsweeps
                % descending, from top to down
                for i=mera.l:-1:2
                    for l=1:hsweeps
                        for k=1:length(mera.layers{i}.W)
                            mera.layers{i}.optimizeW(TOP{i+1},BOTTOM{i},k);
                            rele=[rele norm(mps.contract-mera.contract)/mps_norm];
                        end
                        for k=1:length(mera.layers{i}.U)
                            mera.layers{i}.optimizeU(TOP{i+1},BOTTOM{i},k);
                            rele=[rele norm(mps.contract-mera.contract)/mps_norm];
                        end                        
                    end
                    TOP{i}=mera.layers{i}.topcon(TOP{i+1});
                end                
                % ascending, from bottom to top
                for i=1:mera.l
                    for k=1:length(mera.layers{i}.W)
                        mera.layers{i}.optimizeW(TOP{i+1},BOTTOM{i},k);
                        rele=[rele norm(mps.contract-mera.contract)/mps_norm];
                    end
                    for k=1:length(mera.layers{i}.U)
                        mera.layers{i}.optimizeU(TOP{i+1},BOTTOM{i},k);
                        rele=[rele norm(mps.contract-mera.contract)/mps_norm];
                    end
                    BOTTOM{i+1}=mera.layers{i}.bottomcon(BOTTOM{i});
                end
                % optimize the top tensor
%                 mera.top=lCore([1 1],mera.top.n,reshape(BOTTOM{mera.l+1}.contract/BOTTOM{mera.l+1}.norm*mps_norm,mera.top.n));
                mera.top=lCore([1 1],mera.top.n,reshape(BOTTOM{mera.l+1}.contract,mera.top.n));
                rele=[rele norm(mps.contract-mera.contract)/mps_norm];
                TOP{end}=mera.top.split;
            end
        end
        
        function [mps,e]=mps(mera,epsilon)
            % converts the MERA back into an MPS (with open or closed
            % boundary conditions, dependings on the ranks of the top
            % tensor.
           [mps,e]=mera.top.mps(epsilon);
           for i=mera.l:-1:1
               mps=mera.layers{i}.topcon(mps,epsilon);
           end            
        end
        
        function tensor=contract(mera)
           % contracts the MERA into a N-way tensor where N is the number
           % of bottom legs of the first layer
           tensor=mera.mps(0);
           tensor=tensor.contract;
        end
        
        function n=norm(mera)
           n=mera.top.norm; 
        end
        
        function s=numel(mera)
            s=numel(mera.top.core);
            for i=1:mera.l
               s=s+mera.layers{i}.numel; 
            end
        end
        
    end
    
    
end