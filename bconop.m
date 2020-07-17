function mps2=bconop(layer,mps,epsilon)
% contraction of a open MERA layer at its bottom with an MPS, resulting
% in another MPS that has N/k free indices

% an open MERA has N=p*k (p positive integer) bottom legs and p isometries

% first pair the nodes and contract with the row of disentanglers
% nodes 1,2 go to u_1, 
% nodes k+1,k+2 go to u_2, 
% ....,
% nodes (p-1)*k+1,(p-1)*k+2 go to u_p

N=layer.N;
k=layer.k;
p=N/k;
new_mps=mps.submpt(1,k-1);
for i=1:p-1
    % 2 neighbouring nodes of the MPS are contracted
    temp=mps.subcon(i*k,i*k+1);
    % contract the MPS supercore with the disentangler
    temp=layer.U{i}.bottomcon(temp);
    % split temp supercore into 2 linked cores with svd
    split_temp=temp.mps(epsilon);    
    % concatenate 2 cores with remaining k-2 ones
    if k==2
        new_mps=[new_mps,split_temp];
    else
        new_mps=[new_mps,split_temp,mps.submpt(i*k+2,(i+1)*k-1)];
    end
end
new_mps=[new_mps,mps.submpt(N,N)];

% we now have a new mps that connects directly with the isometries
% now is the time to generate mps2 by contracting the new mps with the
% isometries, keeping the open boundary in mind
cores=cell(1,p);
for i=1:p
    % contract k nodes of the new mps together 
    temp=new_mps.subcon((i-1)*k+1,i*k);
    cores{i}=layer.W{i}.bottomcon(temp);    
end

mps2=MPT(cores);

end     