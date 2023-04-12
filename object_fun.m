function j_fun = object_fun(N,d,k,Cluster_elem,landa,M,fuzzy_degree,W,z,beta_z,p,X,f,b,alpha)
    for j=1:k
        distance(j,:,:) = (1-exp((-1.*repmat(landa,N,1)).*((X-repmat(M(j,:),N,1)).^2)));
        WBETA = transpose(z(j,:).^beta_z);
        WBETA(WBETA==inf)=0;
        dNK(:,j) = reshape(distance(j,:,:),[N,d]) * WBETA * W(1,j)^p ;
    end
    j_fun1 = sum(sum(dNK .* transpose(Cluster_elem.^fuzzy_degree)));
    j_fun2 = sum(sum(dNK .* transpose((Cluster_elem-(b.*f)').^fuzzy_degree)));
    j_fun = j_fun1 +(alpha * j_fun2);
end

