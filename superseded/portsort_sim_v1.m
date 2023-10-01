% Sorted portfolios simulation
% calibration using CZ estimation results
clear;clc;
rng(23,'twister');
R = 1;
T = 60;
N = 100;
J = 10;
K = 5000;
% Lambda
L = gamrnd(2.8,0.026,N,1);
if R==2
    L(:,2) = N*ones(N,1) - N*ones(N,1)'*L(:,1)*L(:,1)/(L(:,1)'*L(:,1));
    L = L/(sqrtm(L'*L));
end
% Mu
% M = [-0.5;-0.4;-0.3;-0.2;-0.1;-0.1;-0.2;-0.3;-0.4;-0.5];
% M = linspace(-0.2,0.8,J)';
% M(10)=-0.2;
M = linspace(-0.5,0.5,J)';
% M(10) = -0.5;
% M = 0.5*ones(J,1);
% M=zeros(J,1);M(1)=-0.5;M(10)=-0.5;
if R==2
    M(:,2) = J*ones(J,1) - J*ones(J,1)'*M(:,1)*M(:,1)/(M(:,1)'*M(:,1));
    M = M/(sqrtm(M'*M));
end
% F
F = -0.032+0.126*randn(T,R);
if R==2
    F = F/(sqrtm(F'*F));
end

s = [216];
Y = ktensor(s,{L,M,F});
Y = tensor(Y);
% sig_u = 1.9; % too large to have decile risk premium pattern
sig_u = 0.5;sig_u1 = 0.9;
U = sig_u*randn([N,J,T]);
% U(:,1,:) = 1.05*U(:,1,:);
% U(:,10,:) = 1.05*U(:,10,:);
for t=1:T
    U(:,1,t) = sig_u1*randn(N,1);
    U(:,10,t) = sig_u1*randn(N,1);
end
Y = double(Y) + U; Y = tensor(Y);
Y = permute(Y,[3,2,1]);
Y_1 = double(reshape(Y,[T,J*N]));
Y_2 = double(reshape(permute(Y,[2,1,3]),[J,T*N]));
Y_3 = double(reshape(permute(Y,[3,1,2]),[N,T*J]));

% tensor decomposition via PCA  (a fair comparison would be tensor power method)
% Lambda
[Gamma_1,S_1] = eig(Y_3*Y_3');
%Gamma_1 = Gamma_1*sqrt(N);
L_hat = NaN(N,R);
for r=1:R
    L_hat(:,r) = Gamma_1(:,N-r+1);
end
s_1 = NaN(R,1);
for r=1:R
    s_1(r) = sqrt(S_1(N-r+1,N-r+1));
end
% Mu
[Gamma_2,S_2] = eig(Y_2*Y_2');
%Gamma_2 = Gamma_2*sqrt(J);
M_hat = NaN(J,R);
for r=1:R
    M_hat(:,r) = Gamma_2(:,J-r+1);
end
s_2 = NaN(R,1);
for r=1:R
    s_2(r) = sqrt(S_2(J-r+1,J-r+1));
end
% F
[Gamma_3,S_3] = eig(Y_1*Y_1');
%Gamma_3 = Gamma_3*sqrt(T);
F_hat = NaN(T,R);
for r=1:R
    F_hat(:,r) = Gamma_3(:,T-r+1);
end
s_3 = NaN(R,1);
for r=1:R
    s_3(r) = sqrt(S_3(T-r+1,T-r+1));
end

weights = M_hat(:,1)-mean(M_hat(:,1));
%% 
pfs = double(Y);
avg=NaN(N,J);
for n=1:N
    avg(n,:) = mean(pfs(:,:,n),1);
end
if sum((mean(avg,1).*weights')>0)<5
    weights = weights*(-1);
end
% risk premium constructed by tensor decomp
RP_ts = NaN(T,N);
for n=1:N
    RP_ts(:,n) = pfs(:,:,n) * weights;
end
hml = [1;zeros(8,1);-1];
RP_hml = NaN(T,N);
for n=1:N
    RP_hml(:,n) = pfs(:,:,n) * hml;
end
stdev_ts = NaN(N,1);stdev_hml = NaN(N,1);
mean_ts = NaN(N,1);mean_hml = NaN(N,1);
for n=1:N
    stdev_ts(n) = std(RP_ts(:,n));
    stdev_hml(n) = std(RP_hml(:,n));
    mean_ts(n) = mean(RP_ts(:,n));
    mean_hml(n) = mean(RP_hml(:,n));
end
diff_std = stdev_hml - stdev_ts;
diff_mean = mean_hml - mean_ts;
size(find(diff_std>0),1)/N
size(find(diff_mean<0),1)/N
sr_ts = mean_ts./stdev_ts;
sr_hml = mean_hml./stdev_hml;
diff_sr = sr_ts - sr_hml;
