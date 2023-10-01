% tensor decomposition via PCA, Chen & Zimmerman factor zoo
% two-way factor model of each decile separately, then average
clear;
rng(23,'twister');
Data1 = readtable('CZ90-20.csv','ReadVariableNames',true);
Date1 = unique(table2array(Data1(:,3)));
N = max(Data1.signum);
J = 10;
T = 360;
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
RF_avg = mean(RF);

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

% tensor decomposition via PCA 
R=5;
% F - unit norm
[Gamma,S] = eig(Y_1*Y_1');
[~,ind] = sort(diag(S),'descend');
F_ts = Gamma(:,ind(1:R)); 

% F - non-unit norm
% [Gamma,S] = eig(Y_1'*Y_1);
% [~,ind] = sort(diag(S),'descend');
% L_ts = Gamma(:,ind(1:R));
% F_ts = Y_1 * L_ts/(L_ts'*L_ts);

% sharpe ratio to approximate SDF, doesnot subtract risk free rate since F
% only identified up to scale
w_ts = cov(F_ts)\mean(F_ts,1)';
port = F_ts*w_ts;
SR_ts = mean(port)/std(port)

a_ts = NaN(J*N,1);
idiovar_ts = NaN(J*N,1);
for n=1:J*N
    [b,~,r] = regress(Y_1(:,n),[ones(T,1),F_ts]);
    a_ts(n) = b(1);
    idiovar_ts(n) = var(r)/var(Y_1(:,n));
end
RMSa_ts = rms(a_ts) % root mean squared pricing error
idiovar_ts = mean(idiovar_ts) % idiosyncratic variation

a_tw = NaN(J*N,J);
idiovar_tw = NaN(J*N,J);
SR_tw = NaN(J,1);
for j=1:J

    Y_tw = permute(Y(:,j,:),[1,3,2]);

    % F_tw
    [Gamma,S] = eig(Y_tw*Y_tw');
    [~,ind] = sort(diag(S),'descend');
    F_tw = Gamma(:,ind(1:R));
    
    w_tw = cov(F_tw)\mean(F_tw,1)';
    port = F_tw*w_tw;
    SR_tw(j) = mean(port)/std(port); % sharpe ratio to approximate SDF
    % for n=1:N
    %     [b,~,r] = regress(Y_tw(:,n),[ones(T,1),F_tw]);
    %     a_tw(n,j) = b(1);
    %     idiovar_tw(n,j) = var(r)/var(Y_tw(:,n));
    % end
    for n=1:J*N
        [b,~,r] = regress(Y_1(:,n),[ones(T,1),F_tw]);
        a_tw(n,j) = b(1);
        idiovar_tw(n,j) = var(r)/var(Y_1(:,n));
    end
end
SR_tw = mean(SR_tw)
RMSa_tw = rms(a_tw,'all')
idiovar_tw = mean(idiovar_tw,'all')


%% out-of-sample
TT = 200;
a_ts_oos = NaN(T-TT,J*N);
a_tw_oos = NaN(T-TT,J*N,J);
port_ts = NaN(T-TT,1);
port_tw = NaN(T-TT,J);
for t=1:(T-TT)
    Y_sub = Y(t:TT+t-1,:,:);
    Y_sub_1 = reshape(Y_sub,[TT,J*N]);
    Y_sub_2 = reshape(permute(Y_sub,[2,1,3]),[J,TT*N]);
    Y_sub_3 = reshape(permute(Y_sub,[3,1,2]),[N,TT*J]);
    
    % tensor decomposition via PCA 
    R=5;
    % Lambda
    [Gamma,S] = eig(Y_sub_1*Y_sub_1');
    [~,ind] = sort(diag(S),'descend');
    F_ts = Gamma(:,ind(1:R));

    beta_ts = Y_sub_1'*F_ts/(F_ts'*F_ts); % cross-sectional asset pricing
    F_ts_oos = regress(Y_1(t+TT,:)',beta_ts);

    w_ts = cov(F_ts)\mean(F_ts,1)'; %optimal portfolio weights
    port_ts(t) = w_ts'*F_ts_oos;
    
    for n=1:J*N
        a_ts_oos(t,n) = Y_1(t+TT,n) - beta_ts(n,:)*F_ts_oos;
    end

    for j=1:J

        Y_tw = permute(Y_sub(:,j,:),[1,3,2]);

        % F_tw
        [Gamma,S] = eig(Y_tw*Y_tw');
        [~,ind] = sort(diag(S),'descend');
        F_tw = Gamma(:,ind(1:R));

        beta_tw = Y_sub_1'*F_tw/(F_tw'*F_tw); % calculate betas
        F_tw_oos = regress(Y_1(t+TT,:)',beta_tw); % predict factors t+1
        
        w_tw = cov(F_tw)\mean(F_tw,1)'; %optimal portfolio weights
        port_tw(t,j) = w_tw'*F_tw_oos;

        for n=1:J*N
            a_tw_oos(t,n,j) = Y_1(t+TT,n) - beta_tw(n,:)*F_tw_oos; % pricing error
        end
    end
end

SR_ts_oos = mean(port_ts)/std(port_ts)
RMSa_ts_oos = rms(a_ts_oos,'all')
idiovar_ts_oos = mean(var(a_ts_oos,0,1)./var(Y_1(TT+1:end,:),0,1))

SR_tw_oos = mean(mean(port_tw,1)./std(port_tw,0,1))
RMSa_tw_oos = rms(a_tw_oos,'all')
idiovar_tw_oos = NaN(J,1);
for j=1:J
    idiovar_tw_oos(j) = mean(var(a_tw_oos(:,:,j),0,1)./var(Y_1(TT+1:end,:),0,1));
end
idiovar_tw_oos = mean(idiovar_tw_oos)
