% tensor decomposition via PCA, Chen & Zimmerman factor zoo
% tensor PCA vs ALS orthogonalized
clear;clc;
rng(23,'twister');
Data1 = readtable('CZ90-20.csv','ReadVariableNames',true);
Date1 = unique(table2array(Data1(:,3)));
N = max(Data1.signum);
J = 10;
T = 360;
R = 5;
NN = N*J*T;
Data2 = readtable('F-F_Research_Data_5_Factors_2x3.csv','ReadVariableNames',true,'VariableNamingRule','preserve');
Data2.Date = datetime(Data2.Date*100+1,'ConvertFrom','yyyymmdd','Format','yyyy-MM-dd');
startyear = 1990;
endyear = 2019;
first2 = find(Data2.Date>=datetime(startyear,01,01),1);
last2 = find(Data2.Date>=datetime(endyear,12,01),1);
mkt = table2array(Data2(first2:last2,"Mkt-RF"));% + table2array(Data2(first2:last2,"RF"));
% HML = table2array(Data2(first2:last2,"HML"));
% RMW = table2array(Data2(first2:last2,"RMW"));
% CMA = table2array(Data2(first2:last2,"CMA"));
RF = table2array(Data2(first2:last2,"RF"));

Y = NaN(T,J,N);
for n=1:N
    Y(:,:,n) = reshape(Data1.ret((n-1)*T*J+1:n*T*J),T,J);
end
% excess return
Y = Y - repmat(RF,1,J,N);

% Y_mkt = NaN(T,J,N);
% for j=1:J
%     for n=1:N
%         [~,~,Y_mkt(:,j,n)] = regress(Y(:,j,n),mkt);
%     end
% end
% Y = Y_mkt;

% avg=NaN(N,J);
% for n=1:N
%     avg(n,:) = mean(Y(:,:,n),1);
% end
% for n=1:N
%     if mean(avg(n,1:5))-mean(avg(n,6:10))>0
%         avg(n,:) = avg(n,J:-1:1);
%         Y(:,:,n) = Y(:,J:-1:1,n);
%     end
% end

% Y = tensor(Y);
Y_1 = reshape(Y,[T,J*N]);
Y_2 = reshape(permute(Y,[2,1,3]),[J,T*N]);
Y_3 = reshape(permute(Y,[3,1,2]),[N,T*J]);

% tensor decomposition via PCA 
% F
[Gamma,S] = eig(Y_1*Y_1');
[s1,ind] = sort(diag(S),'descend');
F_hat = Gamma(:,ind(1:R));
s1 = sqrt(s1(1:R));
% Mu
[Gamma,S] = eig(Y_2*Y_2');
[s2,ind] = sort(diag(S),'descend');
M_hat = Gamma(:,ind(1:R));
s2 = sqrt(s2(1:R));
% Lambda
[Gamma,S] = eig(Y_3*Y_3');
[s3,ind] = sort(diag(S),'descend');
L_hat = Gamma(:,ind(1:R));
s3 = sqrt(s3(1:R));

[~,~,~,~,stats]=regress(F_hat(:,1),mkt)