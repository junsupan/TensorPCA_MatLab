% Sorted portfolios simulation
% calibration

clear;clc;
Data = readtable('CZ96-20.csv','ReadVariableNames',true);
Date = unique(table2array(Data(:,3)));
N = table2array(Data(1,12));
T = 300;
J=10;
rets = table2array(Data(:,4));
Y = NaN(T,J,N);
for n=1:N
    Y(:,:,n) = reshape(rets((n-1)*T*J+1:n*T*J),T,J);
end
avg_data=NaN(N,J);
for n=1:N
    avg_data(n,:) = mean(Y(:,:,n),1);
end
stdev_data=NaN(N,J);
for n=1:N
    stdev_data(n,:) = std(Y(:,:,n),1);
end
for n=1:N
    if mean(avg_data(n,1:5))-mean(avg_data(n,6:10))>0
        avg_data(n,:) = avg_data(n,J:-1:1);
    end
end
mean(avg_data,1)
std(avg_data,1)
mean(stdev_data,1)
std(stdev_data,1)

%% simulation

rng(23,'twister');
avg_sim = repmat(mean(avg_data,1),N,1);
avg_sim = avg_sim + randn(N,J).*repmat(std(avg_data,1),N,1);
stdev_sim = repmat(mean(stdev_data,1),N,1);
stdev_sim = stdev_sim + randn(N,J).*repmat(std(stdev_data,1),N,1);
portfolios_sim = NaN(T,J,N);
for n=1:N
    portfolios_sim(:,:,n) = repmat(avg_sim(n,:),T,1) + randn(T,J).*repmat(stdev_sim(n,:),T,1);
end
Y = NaN(T,J,N);
for n=1:N
    Y(:,:,n) = portfolios_sim(:,:,n) - repmat(mean(portfolios_sim(:,:,n),1),T,1);
end

%% estimation
Y = tensor(Y);
Y_1 = double(reshape(Y,[T,J*N]));
Y_2 = double(reshape(permute(Y,[2,1,3]),[J,T*N]));
Y_3 = double(reshape(permute(Y,[3,1,2]),[N,T*J]));

% tensor decomposition via PCA  (a fair comparison would be tensor power method)
% A
R = 1; % rank
[Gamma_1,S_1] = eig(Y_1*Y_1');
A = NaN(T,R);
for r=1:R
    A(:,r) = Gamma_1(:,T-r+1);
end
s_1 = NaN(R,1);
for r=1:R
    s_1(r) = sqrt(S_1(T-r+1,T-r+1));
end
% B
[Gamma_2,S_2] = eig(Y_2*Y_2');
B = NaN(J,R);
for r=1:R
    B(:,r) = Gamma_2(:,J-r+1);
end
s_2 = NaN(R,1);
for r=1:R
    s_2(r) = sqrt(S_2(J-r+1,J-r+1));
end
% C
[Gamma_3,S_3] = eig(Y_3*Y_3');
C = NaN(N,R);
for r=1:R
    C(:,r) = Gamma_3(:,N-r+1);
end
s_3 = NaN(R,1);
for r=1:R
    s_3(r) = sqrt(S_3(N-r+1,N-r+1));
end

weights = B-mean(B);