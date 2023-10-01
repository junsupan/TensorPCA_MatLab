% tensor decomposition via PCA, Chen & Zimmerman factor zoo
% tensor vs hml vs hal
clear;clc;
rng(23,'twister');
Data1 = readtable('CZ90-20.csv','ReadVariableNames',true);
Date1 = unique(table2array(Data1(:,3)));
N = max(Data1.signum);
J = 10;
T = 360;
R = 3;
Data2 = readtable('F-F_Research_Data_5_Factors_2x3.csv','ReadVariableNames',true,'VariableNamingRule','preserve');
Data2.Date = datetime(Data2.Date*100+1,'ConvertFrom','yyyymmdd','Format','yyyy-MM-dd');
startyear = 1990;
endyear = 2019;
first2 = find(Data2.Date>=datetime(startyear,01,01),1);
last2 = find(Data2.Date>=datetime(endyear,12,01),1);
% mkt = table2array(Data2(first2:last2,"Mkt-RF"));% + table2array(Data2(first2:last2,"RF"));
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

% statistics of the data
% stdev=NaN(N,J);
% for n=1:N
%     stdev(n,:) = std(Y(:,:,n),1);
% end
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
% F
[Gamma,S] = eig(Y_1*Y_1');
[~,ind] = sort(diag(S),'descend');
F_ts = Gamma(:,ind(1:R));

% [Gamma,S] = eig(Y_1'*Y_1);
% [~,ind] = sort(diag(S),'descend');
% L_hat = Gamma(:,ind(1:R));
% 
% r2=NaN(T,1);Y_res = Y_1';
% for n=1:T
%     [~,~,~,~,stats] = regress(Y_res(:,n),[ones(J*N,1),L_hat]);
%     r2(n) = stats(1);
% end
% mean(r2)

% Mu
[Gamma,S] = eig(Y_2*Y_2');
[~,ind] = sort(diag(S),'descend');
M_ts = Gamma(:,ind(1:R));

% Lambda
[Gamma,S] = eig(Y_3*Y_3');
[~,ind] = sort(diag(S),'descend');
L_ts = Gamma(:,ind(1:R));

% F_hml
[Gamma,S] = eig(Y_hml*Y_hml');
[~,ind] = sort(diag(S),'descend');
F_hml = Gamma(:,ind(1:R));

% Lambda_hml
[Gamma,S] = eig(Y_hml'*Y_hml);
[~,ind] = sort(diag(S),'descend');
L_hml = Gamma(:,ind(1:R));

a_ts = NaN(J*N,1);a_hml = NaN(J*N,1);
idiovar_ts = NaN(J*N,1);idiovar_hml = NaN(J*N,1);
for n=1:J*N
    [b,~,r] = regress(Y_1(:,n),[ones(T,1),F_ts]);
    a_ts(n) = b(1);
    idiovar_ts(n) = var(r)/var(Y_1(:,n));
    [b,~,r] = regress(Y_1(:,n),[ones(T,1),F_hml]);
    a_hml(n) = b(1);
    idiovar_hml(n) = var(r)/var(Y_1(:,n));
end
mean(idiovar_ts)
mean(idiovar_hml)

% % a_ts = NaN(N,2);
% a_hml = NaN(N,2);
% % idiovar_ts = NaN(N,2);
% idiovar_hml = NaN(N,2);
% for n=1:N
% %     [b,~,r] = regress(Y(:,1,n),[ones(T,1),F_ts]);
% %     a_ts(n,1) = b(1);
% %     idiovar_ts(n,1) = var(r)/var(Y(:,1,n));
% %     [b,~,r] = regress(Y(:,J,n),[ones(T,1),F_ts]);
% %     a_ts(n,2) = b(1);
% %     idiovar_ts(n,2) = var(r)/var(Y(:,J,n));
%     [b,~,r] = regress(Y(:,1,n),[ones(T,1),F_hml]);
%     a_hml(n,1) = b(1);
%     idiovar_hml(n,1) = var(r)/var(Y(:,1,n));
%     [b,~,r] = regress(Y(:,J,n),[ones(T,1),F_hml]);
%     a_hml(n,2) = b(1);
%     idiovar_hml(n,2) = var(r)/var(Y(:,J,n));
% end
% % mean(idiovar_ts,'all')
% mean(idiovar_hml,'all')


%% out-of-sample
TT = 200;
a_ts_oos = NaN(T-TT,J*N);
a_hml_oos = NaN(T-TT,J*N);
a_hal_oos = NaN(T-TT,J*N);
port_ts = NaN(T-TT,1);
port_hml = NaN(T-TT,1);
port_hal = NaN(T-TT,1);
for t=1:(T-TT)
    Y_sub = Y(t:TT+t-1,:,:);
    Y_sub_1 = reshape(Y_sub,[TT,J*N]);
    Y_sub_2 = reshape(permute(Y_sub,[2,1,3]),[J,TT*N]);
    Y_sub_3 = reshape(permute(Y_sub,[3,1,2]),[N,TT*J]);

    Y_sub_hml = NaN(TT,N);
    for i=1:N
        Y_sub_hml(:,i) = Y_sub(:,J,i) - Y_sub(:,1,i);
    end

    Y_sub_hal = Y(t:TT+t-1,[1,J],:);
    Y_sub_hal = reshape(Y_sub_hal,[TT,2*N]);
    
    % tensor decomposition via PCA 
    % F
    [Gamma,S] = eig(Y_sub_1*Y_sub_1');
    [~,ind] = sort(diag(S),'descend');
    F_ts = Gamma(:,ind(1:R));

    % F_hml
    [Gamma,S] = eig(Y_sub_hml*Y_sub_hml');
    [~,ind] = sort(diag(S),'descend');
    F_hml = Gamma(:,ind(1:R));

    % F_hal
    [Gamma,S] = eig(Y_sub_hal*Y_sub_hal');
    [~,ind] = sort(diag(S),'descend');
    F_hal = Gamma(:,ind(1:R));
    
%     beta_ts = NaN(J*N,R);beta_hml = NaN(J*N,R);
%     for n=1:J*N
%         b = regress(Y_sub_1(:,n),[ones(TT,1),F_hat]);
%         beta_ts(n,:) = b(2:end);
%         b = regress(Y_sub_1(:,n),[ones(TT,1),F_hat_hml]);
%         beta_hml(n,:) = b(2:end);
%     end

    beta_ts = Y_sub_1'*F_ts/(F_ts'*F_ts);
    beta_hml = Y_sub_1'*F_hml/(F_hml'*F_hml);
    beta_hal = Y_sub_1'*F_hal/(F_hal'*F_hal);

    F_ts_oos = regress(Y_1(t+TT,:)',beta_ts);
    F_hml_oos = regress(Y_1(t+TT,:)',beta_hml);
    F_hal_oos = regress(Y_1(t+TT,:)',beta_hal);
    for n=1:J*N
        a_ts_oos(t,n) = Y_1(t+TT,n) - beta_ts(n,:)*F_ts_oos;
        a_hml_oos(t,n) = Y_1(t+TT,n) - beta_hml(n,:)*F_hml_oos;
        a_hal_oos(t,n) = Y_1(t+TT,n) - beta_hal(n,:)*F_hal_oos;
    end

    w_ts = cov(F_ts)\mean(F_ts,1)'; %optimal portfolio weights
    port_ts(t) = w_ts'*F_ts_oos;

    w_hml = cov(F_hml)\mean(F_hml,1)'; %optimal portfolio weights
    port_hml(t) = w_hml'*F_hml_oos;

    w_hal = cov(F_hal)\mean(F_hal,1)'; %optimal portfolio weights
    port_hal(t) = w_hal'*F_hal_oos;
end
RMSa_ts_oos = mean(rms(a_ts_oos,1))
RMSa_hml_oos = mean(rms(a_hml_oos,1))
RMSa_hal_oos = mean(rms(a_hal_oos,1))
idiovar_ts_oos = mean(var(a_ts_oos,0,1)./var(Y_1(TT+1:end,:),0,1))
idiovar_hml_oos = mean(var(a_hml_oos,0,1)./var(Y_1(TT+1:end,:),0,1))
idiovar_hal_oos = mean(var(a_hal_oos,0,1)./var(Y_1(TT+1:end,:),0,1))
SR_ts_oos = mean(port_ts)/std(port_ts)
SR_hml_oos = mean(port_hml)/std(port_hml)
SR_hal_oos = mean(port_hal)/std(port_hal)
