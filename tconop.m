function mps2=tconop(layer,mps,epsilon)
% contraction of an open MERA layer at its top with an MPS, resulting
% in another MPS that has N*k free indices

% an open MERA has N=p*k bottom legs and p isometries,

N=layer.N;
k=layer.k;
p=N/k;
new_mps=[];
for i=1:p
    % contract each MPS core with the corresponding isometry
    temp=layer.W{i}.topcon(mps.cores{i});
    % split temp supercore into 2 linked cores with svd
    split_temp=temp.mps(epsilon);  
    % concatenate 2 cores with remaining k-2 ones
    new_mps=[new_mps,split_temp];
end

% we now have a new mps that connects directly with the disentanglers
% now is the time to generate mps2 by contracting the new mps with the
% disentanglers, keeping the open boundary in mind
mps2=new_mps.submpt(1,k-1);
for i=1:p-1
    % contract 2 nodes of the new mps with the disentangler 
    temp=new_mps.subcon(i*k,i*k+1);
    temp=layer.U{i}.topcon(temp);
    temp=temp.mps(epsilon); % split the Core into an mps
    if k==2
        mps2=[mps2,temp];
    else
        mps2=[mps2,temp,MPT(new_mps.cores(i*k+2:(i+1)*k-1))];
    end
end
% append final Core
mps2=[mps2,new_mps.submpt(N,N)];

end     