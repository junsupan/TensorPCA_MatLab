% tensor decomposition via PCA, Chen & Zimmerman factor zoo
% beta_ij = lambda_i * mu_j
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

% Y = tensor(Y);
Y_1 = reshape(Y,[T,J*N]);
Y_2 = reshape(permute(Y,[2,1,3]),[J,T*N]);
Y_3 = reshape(permute(Y,[3,1,2]),[N,T*J]);
% F
[Gamma,S] = eig(Y_1*Y_1');
[~,ind] = sort(diag(S),'descend');
F_ts = Gamma(:,ind(1:R));

% Mu
[Gamma,S] = eig(Y_2*Y_2');
[~,ind] = sort(diag(S),'descend');
M_ts = Gamma(:,ind(1:R));

% Lambda
[Gamma,S] = eig(Y_3*Y_3');
[~,ind] = sort(diag(S),'descend');
L_ts = Gamma(:,ind(1:R));

beta_ts = NaN(J*N,R);
for r=1:R
    beta_ts(:,r) = reshape(M_ts(:,r)*L_ts(:,r)',[J*N,1]);
end
beta_ts = beta_ts/(sqrtm(beta_ts'*beta_ts));

% beta two-way
[Gamma,S] = eig(Y_1'*Y_1);
[~,ind] = sort(diag(S),'descend');
beta_tw = Gamma(:,ind(1:R));

F_ts = Y_1*beta_ts/(beta_ts'*beta_ts);
F_tw = Y_1*beta_tw/(beta_tw'*beta_tw);

w_ts = cov(F_ts)\mean(F_ts,1)';
port = F_ts*w_ts;
SR_ts = mean(port)/std(port)

w_tw = cov(F_tw)\mean(F_tw,1)';
port = F_tw*w_tw;
SR_tw = mean(port)/std(port)

a_ts = NaN(J*N,1);
idiovar_ts = NaN(J*N,1);
a_tw = NaN(J*N,1);
idiovar_tw = NaN(J*N,1);
for n=1:J*N
    [b,~,r] = regress(Y_1(:,n),[ones(T,1),F_ts]);
    a_ts(n) = b(1);
    idiovar_ts(n) = var(r)/var(Y_1(:,n));
    [b,~,r] = regress(Y_1(:,n),[ones(T,1),F_tw]);
    a_tw(n) = b(1);
    idiovar_tw(n) = var(r)/var(Y_1(:,n));
end
RMSa_ts = rms(a_ts) % root mean squared pricing error
idiovar_ts = mean(idiovar_ts) % idiosyncratic variation
RMSa_tw = rms(a_tw) % root mean squared pricing error
idiovar_tw = mean(idiovar_tw) % idiosyncratic variation


%% out-of-sample
TT = 200;
a_ts_oos = NaN(T-TT,J*N);
a_tw_oos = NaN(T-TT,J*N);
port_ts = NaN(T-TT,1);
port_tw = NaN(T-TT,1);
for t=1:(T-TT)
    Y_sub = Y(t:TT+t-1,:,:);
    Y_sub_1 = reshape(Y_sub,[TT,J*N]);
    Y_sub_2 = reshape(permute(Y_sub,[2,1,3]),[J,TT*N]);
    Y_sub_3 = reshape(permute(Y_sub,[3,1,2]),[N,TT*J]);
    
    % tensor decomposition via PCA 
    % F
    [Gamma,S] = eig(Y_sub_1*Y_sub_1');
    [~,ind] = sort(diag(S),'descend');
    F_ts = Gamma(:,ind(1:R));
    
    % Mu
    [Gamma,S] = eig(Y_sub_2*Y_sub_2');
    [~,ind] = sort(diag(S),'descend');
    M_ts = Gamma(:,ind(1:R));

    % Lambda
    [Gamma,S] = eig(Y_sub_3*Y_sub_3');
    [~,ind] = sort(diag(S),'descend');
    L_ts = Gamma(:,ind(1:R));
    
    % two-way beta
    [Gamma,S] = eig(Y_sub_1'*Y_sub_1);
    [~,ind] = sort(diag(S),'descend');
    beta_tw = Gamma(:,ind(1:R));

    beta_ts = NaN(J*N,R);
    for r=1:R
        beta_ts(:,r) = reshape(M_ts(:,r)*L_ts(:,r)',[J*N,1]);
    end
    beta_ts = beta_ts/(sqrtm(beta_ts'*beta_ts));

    F_ts_oos = regress(Y_1(t+TT,:)',beta_ts);
    F_tw_oos = regress(Y_1(t+TT,:)',beta_tw);
    for n=1:J*N
        a_ts_oos(t,n) = Y_1(t+TT,n) - beta_ts(n,:)*F_ts_oos;
        a_tw_oos(t,n) = Y_1(t+TT,n) - beta_tw(n,:)*F_tw_oos;
    end
    
    F_ts = Y_sub_1*beta_ts/(beta_ts'*beta_ts);
    w_ts = cov(F_ts)\mean(F_ts,1)'; %optimal portfolio weights
    port_ts(t) = w_ts'*F_ts_oos;
    
    F_tw = Y_sub_1*beta_tw/(beta_tw'*beta_tw);
    w_tw = cov(F_tw)\mean(F_tw,1)'; %optimal portfolio weights
    port_tw(t) = w_tw'*F_tw_oos;
end
RMSa_ts_oos = rms(a_ts_oos,'all')
RMSa_tw_oos = rms(a_tw_oos,'all')
idiovar_ts_oos = mean(var(a_ts_oos,0,1)./var(Y_1(TT+1:end,:),0,1))
idiovar_tw_oos = mean(var(a_tw_oos,0,1)./var(Y_1(TT+1:end,:),0,1))
SR_ts_oos = mean(port_ts)/std(port_ts)
SR_tw_oos = mean(port_tw)/std(port_tw)
