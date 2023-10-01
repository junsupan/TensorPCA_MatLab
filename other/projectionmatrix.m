clear;clc;
rng(23,'twister');
Pi = [];
Ns = 50:50:500;
R = 2;
for N = Ns
    a = randn(N,R);
    A = a * a';
    [V,D] = eig(A);
    Vr = V(:,1:N-R);
    P = Vr/(Vr'*Vr)*Vr';
    Pi = [Pi,sum(diag(P).^2)/N];
end
plot(Ns,Pi)