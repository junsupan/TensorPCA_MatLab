% Chen & Zimmerman factor zoo, test # of factors
clear;clc;
rng(23,'twister');
Data1 = readtable('CZ90-20.csv','ReadVariableNames',true);
Date1 = unique(table2array(Data1(:,3)));
N = max(Data1.signum);
J = 10;
T = 360;
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

Y_mkt = NaN(T,J,N);
for j=1:J
    for n=1:N
        [~,~,Y_mkt(:,j,n)] = regress(Y(:,j,n),mkt);
    end
end
Y = Y_mkt;

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
% F_hat = Gamma(:,ind(1:R));
% s1 = sqrt(s1(1:R));
% Mu
[Gamma,S] = eig(Y_2*Y_2');
[s2,ind] = sort(diag(S),'descend');
% M_hat = Gamma(:,ind(1:R));
% s2 = sqrt(s2(1:R));
% Lambda
[Gamma,S] = eig(Y_3*Y_3');
[s3,ind] = sort(diag(S),'descend');
% L_hat = Gamma(:,ind(1:R));
% s3 = sqrt(s3(1:R));

R_list = 1:4;
p1 = nan(length(R_list),1);p2 = nan(length(R_list),1);p3 = nan(length(R_list),1);
for i=1:length(R_list)
R = R_list(i);disp(i);
% approximate distribution
K0 = R+1;% parameter in TW distribution
K = 5000;
asydist = NaN(K,1);
for k=1:K
    if mod(k, 100) == 0
        disp(k);
    end
%     Z = randn([min(T,N)-R,min(T,N)-R]);
%     Z = tril(Z,-1);
%     Z = Z + Z';
%     Z(1:min(T,N)-R+1:end) = sqrt(2)*randn(min(T,N)-R,1);
    
    Z = randn([1000,1000]);
    Z = tril(Z,-1);
    Z = Z + Z';
    Z(1:1001:end) = sqrt(2)*randn(1000,1);

    [~,s_z] = eig(Z);
    s_z = sort(diag(s_z),'descend');
%     asydist(m) = s_z(1)-s_z(end);
    
    asydist_i = NaN(K0-R,1);
    for r=1:K0-R
        asydist_i(r) = (s_z(r)-s_z(r+1))/(s_z(r+1)-s_z(r+2));
    end
    asydist(k) = max(asydist_i);
end
c = quantile(asydist,0.95);
% c = 11.9853;

% statistic
Sk_i = NaN(K0-R,1);
for r=1:K0-R
    Sk_i(r) = (s1(R+r)-s1(R+r+1))/(s1(R+r+1)-s1(R+r+2));
end
Sk1 = max(Sk_i);

Sk_i = NaN(K0-R,1);
for r=1:K0-R
    Sk_i(r) = (s2(R+r)-s2(R+r+1))/(s2(R+r+1)-s2(R+r+2));
end
Sk2 = max(Sk_i);

Sk_i = NaN(K0-R,1);
for r=1:K0-R
    Sk_i(r) = (s3(R+r)-s3(R+r+1))/(s3(R+r+1)-s3(R+r+2));
end
Sk3 = max(Sk_i);

% p-values
p1(i) = sum(asydist>Sk1)/K;
p2(i) = sum(asydist>Sk2)/K;
p3(i) = sum(asydist>Sk3)/K;

% disp([p1,p2,p3]);

end

%%
table = array2table([p1,p2,p3],'RowNames',{'R=1','R=2','R=3','R=4'},'VariableNames',{'mat1-T','mat2-J','mat3-N'});

%%
plot(1:10,s1(1:10))
hold on;
plot(1:10,s2(1:10))
plot(1:10,s3(1:10))
xlabel('nth factor')
ylabel('eigenvalues')
legend('mat1','mat2','mat3')
