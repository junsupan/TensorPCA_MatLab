function [moms_out, h] = moms_tensor(para, X, K)
% GMM estimation for tensor

[T,N] = size(X);
Lambda = reshape(para,[N,K]);

for j=1:K
h(:,(j-1)*N+1:j*N) = repmat((X*Lambda(:,j)).^2,[1,N]).*X ...
    + repmat(sum(repmat((Lambda'*Lambda(:,j)).^2,[1,N]).*Lambda',1),[T,1]);
end

moms_out = mean(h,1)';
