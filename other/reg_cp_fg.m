function [f,G] = reg_cp_fg(W,A,Wnormsqr,X,lambda)
%calculates function and gradient of regularized CP
% W is investor x firm x time, X is time x firm
% A is cell of matrix arrays

T = size(W,3);

[f,G] = tt_cp_fg(W,A,Wnormsqr);
f = f + lambda/T*norm(X-A{3}*A{2}');
G{2} = G{2} + 2*lambda/T*((A{3}*A{2}')'*A{3} - X'*A{3});
G{3} = G{3} + 2*lambda/T*((A{3}*A{2}')*A{2} - X*A{2});

end