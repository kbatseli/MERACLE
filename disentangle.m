function [Uhat,svals,procruste,maxRhat]=disentangle(mps,index,norm_index,terms,rankgap,MAXITR)
% Tries to find an orthogonal disentangler that covers the cores INDEX and
% INDEX+1 of a given MPS turned into site-NORM_INDEX-mixed-canonical-from.
% The iterative procrustes algorithm constructs a low-rank approximation
% using the rank-1 TERMS. The iterations stop after a certain fixed number
% or when the disentanglement has achieved a certain RANKGAP.
%

% MAXITR=1000;

mps.skmc(norm_index);
supercore=mps.subcon(index,index+1);
% A23 is full-rank matrix,
Aft=supercore.firstk(2); 
% permutation of the first 2 indices of A23 does not influence the rank 
Aftp=reshape(permute(supercore.core,[2,1,3,4]),size(Aft));

Bft=supercore.top2bottom';
[Utest,Stest,Vtest]=svd(Aftp,'econ');
sig{1}=diag(Stest);
tol=sig{1}(1)*eps*max(size(Bft));
origR=mps.r(index+1);              % original TT-rank
temp=Bft;
Uhat=1;
procruste=0;

itr=2;
while itr<MAXITR
%     pick rank-1 terms as right-hand-side of orthogonal Procrustes problem 
    B=Utest(:,terms)*Stest(terms,terms)*Vtest(:,terms)';
    
    B=reshape(B,[supercore.n(1),supercore.r(1),supercore.n(2),supercore.r(2)]);
    B=permute(B,[1,3,2,4]);
    B=reshape(B,[supercore.n(1)*supercore.n(2),supercore.r(1)*supercore.r(2)]);    
   
%     B=Stest(term,term)*kron(reshape(Vtest(:,term),[supercore.n(2),supercore.r(2)]),reshape(Utest(:,term),[supercore.n(1),supercore.r(1)]));

    % procrustus R1 such that R1*B1 = B23
    [Utest,Stest,Vtest]=svd(B*temp','econ');
    R=Utest*Vtest';
    temp=R*temp;
    Uhat=R*Uhat;
    procruste(itr,1)=norm(temp-B,'fro')/norm(B,'fro');
        
    [Utest,Stest,Vtest]=svd(reshape(permute(reshape(temp,[supercore.n(1),supercore.n(2),supercore.r(1),supercore.r(2)]),[1,3,2,4]),[supercore.n(1)*supercore.r(1),supercore.n(2)*supercore.r(2)]),'econ');
    sig{itr}=diag(Stest);
    
    [y,maxRhat]=min(diff(log10(sig{itr})));
    if sig{itr}(maxRhat)/sig{itr}(maxRhat+1) > rankgap && maxRhat < origR
        break
    end
    itr=itr+1;
end

shat=svd(reshape(permute(reshape(Uhat*Bft,[mps.n{index},mps.n{index+1},mps.r(index),mps.r(index+2)]),[1,3,2,4]),[mps.n{index}*mps.r(index),mps.n{index+1}*mps.r(index+2)]),'econ');
[y,maxRhat]=min(diff(log10(shat)));
if shat(maxRhat)/shat(maxRhat+1) < rankgap
    maxRhat=sum(shat>tol); % numerical rank after applying the known disentangler
end
if maxRhat < origR
    disp(['Reduced rank from ' num2str(origR) ' to ' num2str(maxRhat) '.'])
elseif maxRhat  > origR
    disp(['Increased rank from ' num2str(origR) ' to ' num2str(maxRhat) '.'])    
end
    
svals=zeros(mps.n{index+1}*mps.r(index+2),size(sig,2));for i=1:length(sig),svals(1:length(sig{1,i}),i)=sig{1,i}(:);end

% figure
% % subplot(1,2,1)
% semilogy(svals','-o')
% grid on
% subplot(1,2,2)
% semilogy(procruste,'-o')

end