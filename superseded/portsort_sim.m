% Sorted portfolios simulation
% calibration using CZ estimation results
clear;
rng(23,'twister');
R = 1;
T = 60;
N = 100;
J = 10;
K = 5000;
% Lambda
L = -gamrnd(2.8,0.026,N,1);
L = L/(sqrt(L'*L));
% Mu
M = linspace(-0.2,0.65,J)';
M = M/(sqrtm(M'*M));
% F
F = 0.0648+0.0489*randn(T,R);
F = F/(sqrtm(F'*F));

% % Lambda
% L = -gamrnd(0.2565,0.1517,N,1)+0.0096;
% L = L/(sqrt(L'*L));
% % Mu
% M = linspace(-0.2,0.65,J)';
% M = M/(sqrtm(M'*M));
% % F
% F = 0.0019+0.1302*randn(T,R);
% F = F/(sqrtm(F'*F));

% % Lambda
% L = gamrnd(5.046,0.0345,N,1)-0.1476;
% L = L/(sqrt(L'*L));
% % Mu
% M = linspace(-0.35,0.69,J)';
% M = M/(sqrtm(M'*M));
% % F
% F = 0.0149+0.1293*randn(T,R);
% F = F/(sqrtm(F'*F));

% % Lambda
% L = gamrnd(2.6467,0.0301,N,1)-0.0149;
% L = L/(sqrt(L'*L));
% % Mu
% M = linspace(-0.78,0.1,J)';
% M = M/(sqrtm(M'*M));
% % F
% F = 0.0542+0.1181*randn(T,R);
% F = F/(sqrtm(F'*F));

% % Lambda
% L = -gamrnd(4.5733,0.0214,N,1)+0.0284;
% L = L/(sqrt(L'*L));
% % Mu
% M = linspace(-0.4,0.7,J)';
% M = M/(sqrtm(M'*M));
% % F
% F = -0.0373+0.1246*randn(T,R);
% F = F/(sqrtm(F'*F));

s = [300];
Y = ktensor(s,{L,M,F});
Y = tensor(Y);
% sig_u = 1.68; % too large to have decile risk premium pattern
sig_u = 0.5;sig_u1 = 0.5;
U = sig_u*randn([N,J,T]);
for t=1:T
    U(:,1,t) = sig_u1*randn(N,1);
    U(:,10,t) = sig_u1*randn(N,1);
end
Y = double(Y) + U; Y = tensor(Y);
Y = permute(Y,[3,2,1]);
Y_1 = double(reshape(Y,[T,J*N]));
Y_2 = double(reshape(permute(Y,[2,1,3]),[J,T*N]));
Y_3 = double(reshape(permute(Y,[3,1,2]),[N,T*J]));

% tensor decomposition via PCA 
% Lambda
[Gamma_1,S_1] = eig(Y_1*Y_1');
% Gamma_1 = Gamma_1*sqrt(N);
[s_1,ind] = sort(diag(S_1),'descend');
F_hat = Gamma_1(:,ind(1:R));
% Mu
[Gamma_2,S_2] = eig(Y_2*Y_2');
% Gamma_2 = Gamma_2*sqrt(J);
[s_2,ind] = sort(diag(S_2),'descend');
M_hat = Gamma_2(:,ind(1:R));
% F
[Gamma_3,S_3] = eig(Y_3*Y_3');
% Gamma_3 = Gamma_3*sqrt(T);
[s_3,ind] = sort(diag(S_3),'descend');
L_hat = Gamma_3(:,ind(1:R));

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
std_ts = NaN(N,1);std_hml = NaN(N,1);
mean_ts = NaN(N,1);mean_hml = NaN(N,1);
for n=1:N
    std_ts(n) = std(RP_ts(:,n));
    std_hml(n) = std(RP_hml(:,n));
    mean_ts(n) = mean(RP_ts(:,n));
    mean_hml(n) = mean(RP_hml(:,n));
end
ratio_std = std_ts./std_hml;
diff_mean = mean_ts - mean_hml;
r1 = size(find(ratio_std<1),1)/N
r2 = size(find(diff_mean>0),1)/N
sr_ts = (mean_ts)./std_ts;
sr_hml = (mean_hml)./std_hml;
ratio_sr = sr_ts./sr_hml;

scatter(sr_hml,sr_ts);
hold on
sl = (min([sr_hml,sr_ts],[],'all')-0.1):0.1:(max([sr_hml,sr_ts],[],'all')+0.1);
plot(sl,sl);
xlabel('SR\_HML','FontSize',12)
ylabel('SR\_TS','FontSize',12)
title('s_{u1}=1','FontSize',14)
% saveTightFigure('srsim1.pdf')