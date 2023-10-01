% tensor decomposition via PCA, Chen & Zimmerman factor zoo
% no regularization
clear;
top = cell(10,4);
rng(23,'twister');
for sample = 1
range = {'96-00','01-05','06-10','11-15','16-20'};
Data1 = readtable(strcat('CZ',range{sample},'.csv'),'ReadVariableNames',true);
Date = unique(table2array(Data1(:,3)));
N = table2array(Data1(1,12));
Num(sample) = N;
J = 10;
T = 60;
rets = table2array(Data1(:,4));
Data2 = readtable('F-F_Research_Data_5_Factors_2x3.csv','ReadVariableNames',true,'VariableNamingRule','preserve');
Data2.Date = datetime(Data2.Date*100+1,'ConvertFrom','yyyymmdd','Format','yyyy-MM-dd');
startyear = [1996,2001,2006,2011,2016];
endyear = [2000,2005,2010,2015,2020];
first2 = find(Data2.Date>=datetime(startyear(sample),01,01),1);
last2 = find(Data2.Date>=datetime(endyear(sample),12,01),1);
mkt = table2array(Data2(first2:last2,"Mkt-RF"));% + table2array(Data2(first2:last2,"RF"));
HML = table2array(Data2(first2:last2,"HML"));
RMW = table2array(Data2(first2:last2,"RMW"));
CMA = table2array(Data2(first2:last2,"CMA"));
RF = table2array(Data2(first2:last2,"RF"));

Y = NaN(T,J,N);
for n=1:N
    Y(:,:,n) = reshape(rets((n-1)*T*J+1:n*T*J),T,J);
end
avg=NaN(N,J);
for n=1:N
    avg(n,:) = mean(Y(:,:,n),1);
end
for n=1:N
    if mean(avg(n,1:5))-mean(avg(n,6:10))>0
        avg(n,:) = avg(n,J:-1:1);
        Y(:,:,n) = Y(:,J:-1:1,n);
    end
end
stdev=NaN(N,J);
for n=1:N
    stdev(n,:) = std(Y(:,:,n),1);
end
% statistics of the data
% mean_mean(sample,:) = mean(avg,1);
% mean_stdev(sample,:) = std(avg,1);
% stdev_mean(sample,:) = mean(stdev,1);
% stdev_stdev(sample,:) = std(stdev,1);
%
% Y = tensor(Y);
Y_1 = reshape(Y,[T,J*N]);
Y_2 = reshape(permute(Y,[2,1,3]),[J,T*N]);
Y_3 = reshape(permute(Y,[3,1,2]),[N,T*J]);

Y_hml = NaN(T,N);
for i=1:N
    Y_hml(:,i) = Y(:,J,i) - Y(:,1,i);
end

% tensor decomposition via PCA 
R=3;
% F
[Gamma_1,S_1] = eig(Y_1*Y_1');
[s_1,ind] = sort(diag(S_1),'descend');
F_hat = Gamma_1(:,ind(1:R));

% Mu
[Gamma_2,S_2] = eig(Y_2*Y_2');
[s_2,ind] = sort(diag(S_2),'descend');
M_hat = Gamma_2(:,ind(1:R));

% Lambda
[Gamma_3,S_3] = eig(Y_3*Y_3');
[s_3,ind] = sort(diag(S_3),'descend');
L_hat = Gamma_3(:,ind(1:R));

% F_hml
[Gamma_1,S_1] = eig(Y_hml*Y_hml');
[s_1,ind] = sort(diag(S_1),'descend');
F_hat_hml = Gamma_1(:,ind(1:R));

a_ts = NaN(J*N,1);a_hml = NaN(J*N,1);
idiovar_ts = NaN(J*N,1);idiovar_hml = NaN(J*N,1);
for n=1:J*N
    mdl = fitlm(F_hat,Y_1(:,n));
    a_ts(n) = table2array(mdl.Coefficients('(Intercept)','Estimate'));
    idiovar_ts(n) = var(mdl.Residuals.Raw)/var(Y_1(:,n));
    mdl = fitlm(F_hat_hml,Y_1(:,n));
    a_hml(n) = table2array(mdl.Coefficients('(Intercept)','Estimate'));
    idiovar_hml(n) = var(mdl.Residuals.Raw)/var(Y_1(:,n));
end



end
