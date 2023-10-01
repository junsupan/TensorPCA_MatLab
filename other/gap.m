%% E(UU')
clear;
M2=5000;
N1=10;N2=10;s=NaN(5000,1);
for m=1:M2
U=randn(N1,N2);
[~,D]=eig(U*U');
s(m)=max(diag(D));
end
a1=mean(s);
N1=10;N2=1000;s=NaN(5000,1);
for m=1:M2
U=randn(N1,N2);
[~,D]=eig(U*U');
s(m)=max(diag(D));
end
a2=mean(s);
a2/a1


%%
clear;
alpha=1/3;
N1=100;N2=100;N3=100;
a1=(sqrt(N1)*sqrt(N1*N2*N3)^alpha+sqrt(N1*N2*N3)+N1+N2*N3)/(N1*N2*N3)^alpha;
N1=200;N2=200;N3=200;
a2=(sqrt(N1)*sqrt((N1*N2*N3)^alpha)+sqrt(N1*N2*N3)+N1+N2*N3)/(N1*N2*N3)^alpha;
a2/a1

%%
clear;
R = 1;
alpha = 0.5;
M = 1000;
N1 = 50;
N2 = 50;
% generate Lambda
M1 = get_orthonormal(N1,R);
M1_norm = M1/(sqrtm(M1'*M1));

% generate Mu
M2 = get_orthonormal(N2,R);
M2_norm = M2/(sqrtm(M2'*M2));

s = [2*sqrt((N1*N2)^alpha)];
D = diag(s);
sig_u = 0.1;

op = NaN(M,1);
for m = 1:M;
U = sig_u*randn(N1,N2);
Y = M1*D*M2' + U;
diff = Y*Y' - M1*D*D*M1';
% [~,S]=eig(diff);
op(m) = norm(diff);
end
a1 = mean(op)/s^2;

N1 = 100;
N2 = 100;
% generate Lambda
M1 = get_orthonormal(N1,R);
M1_norm = M1/(sqrtm(M1'*M1));

% generate Mu
M2 = get_orthonormal(N2,R);
M2_norm = M2/(sqrtm(M2'*M2));

s = [2*sqrt((N1*N2)^alpha)];
D = diag(s);
sig_u = 0.1;

op = NaN(M,1);
for m = 1:M;
U = sig_u*randn(N1,N2);
Y = M1*D*M2' + U;
diff = Y*Y' - M1*D*D*M1';
% [~,S]=eig(diff);
op(m) = norm(diff);
end
a2 = mean(op)/s^2;

a2/a1

%%
N=100:100:10000;
for i=1:size(N,2);
m=randn(N(i),1);
m=m/norm(m);
n(i)=norm(m,3)^3;
end
plot(n)
