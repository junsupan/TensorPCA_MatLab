addpath(genpath('tensor_toolbox-v3.2.1'))
% rolling sample estimation, only for monthly data
clear;clc;rng(22,'twister');
daily = 0; % 1 for daily, 0 for monthly
p = 49; % number of portfolios, 5 or 49
datetime.setDefaultFormats('defaultdate','yyyy-MM-dd');
if daily == 1
Data = readtable(sprintf('%d_Industry_Portfolios_Daily.csv',p),'ReadVariableNames',true);
Date = datetime(Data.Date,'ConvertFrom','yyyymmdd','Format','yyyy-MM-dd');
startyear = 2000;
Y = readtable('F-F_Research_Data_5_Factors_2x3_daily.csv','ReadVariableNames',true);
DateY = datetime(Y.Date,'ConvertFrom','yyyymmdd','Format','yyyy-MM-dd');
elseif daily == 0
Data = readtable(sprintf('%d_Industry_Portfolios.csv',p),'ReadVariableNames',true);
Date = datetime(Data.Date*100+1,'ConvertFrom','yyyymmdd','Format','yyyy-MM-dd');
startyear = 1980;
Y = readtable('F-F_Research_Data_5_Factors_2x3.csv','ReadVariableNames',true);
DateY = datetime(Y.Date*100+1,'ConvertFrom','yyyymmdd','Format','yyyy-MM-dd');
else
    error('daily is either 1 or 0')
end
first = find(Date>datetime(startyear,01,01),1);
last = find(Date>datetime(2021,06,29),1);
X = table2array(Data);
X = X(first:last,2:end);
assets = X;
firstY = find(DateY>datetime(startyear,01,01),1);
lastY = find(DateY>datetime(2021,06,29),1);
Y = table2array(Y);
Y = Y(firstY:lastY,2:end);

Mkt = Y(:,1)+Y(:,6);

K = 2;
T = size(X,1);
N = size(X,2);
M = 5000;

mu=mean(X,1);
X=X-ones(T,1)*mu; %demeaned

for t=1:T-240
    
    if mod(t, 10) == 0
        disp(t);
    end

    Xroll = X(t:239+t,:);
    %PCA
    Cov=Xroll'*Xroll/240;
    [V,D] = eig(Cov);
    Lambda_pca = NaN(K,N);
    for k=1:K
        Lambda_pca(k,:) = V(:,N-k+1)';
        spec_pca(k) = D(N-k+1,N-k+1);
    end

    %ALS
    skew=NaN(N,N,N);
    for i=1:N
        for j=1:N
            for k=1:N
                skew(i,j,k)=sum(Xroll(:,i).*Xroll(:,j).*Xroll(:,k))/240;
            end
        end
    end
    skew_copy = skew;
    skew = tensor(skew);
    spec_als_copy = [];Lambda_als_copy = [];
    for m=1:M
        CPALS = cp_als(skew,K,'printitn',0);
        spec_als = CPALS.lambda;
        spec_als_copy(m,:) = spec_als;
        Lambda_als = CPALS.U{1}';
        Lambda_als_copy(:,:,m) = Lambda_als;
    end

    ind = 0;
    obj0 = inf;
    for m=1:M
        spec_als = spec_als_copy(m,:).^(1/3);
        Lambda_als = Lambda_als_copy(:,:,m);
        for k=1:K
            Lambda_als(k,:) = Lambda_als(k,:)*spec_als(k);
        end
        Gam=NaN(N,N,N);
        for i=1:N
            for j=1:N
                for k=1:N
                    Gam(i,j,k)=sum(Lambda_als(:,i).*Lambda_als(:,j).*Lambda_als(:,k));
                end
            end
        end
        obj = sum((skew_copy-Gam).^2,'all');
        if obj < obj0
            obj0 = obj;
            ind = m;
        end
    end
    
    Lambda_als = Lambda_als_copy(:,:,ind);
    spec_als = spec_als_copy(ind,:);
    spec_als = spec_als.^(1/3);

    % unit loadings to real loadings
    for k=1:K
        Lambda_als(k,:) = Lambda_als(k,:)*spec_als(k);
    end
    for k=1:K
        Lambda_pca(k,:) = Lambda_pca(k,:)*spec_pca(k);
    end

    % recover factor process
    F_als=assets(t:239+t,:)*Lambda_als'*pinv(Lambda_als*Lambda_als');
    F_pca=assets(t:239+t,:)*Lambda_pca'*pinv(Lambda_pca*Lambda_pca');

    % out-of-sample return
    w_als = inv(cov(F_als))*mean(F_als,1)';
    w_pca = inv(cov(F_pca))*mean(F_pca,1)';
    F_als_pred = assets(240+t,:)*Lambda_als'*pinv(Lambda_als*Lambda_als');
    F_pca_pred = assets(240+t,:)*Lambda_pca'*pinv(Lambda_pca*Lambda_pca');
    r_als(t) = F_als_pred * w_als;
    r_pca(t) = F_pca_pred * w_pca;

    for i=1:N
        mdl = fitlm(F_als,assets(t:239+t,i)-Y(t:239+t,6));
        alpha_als(t,i) = predict(mdl,F_als_pred) - (assets(240+t,i)-Y(240+t,6));
    end
    for i=1:N
        mdl = fitlm(F_pca,assets(t:239+t,i)-Y(t:239+t,6));
        alpha_pca(t,i) = predict(mdl,F_pca_pred) - (assets(240+t,i)-Y(240+t,6));
    end

end
SR_als = mean(r_als)/std(r_als)
SR_pca = mean(r_pca)/std(r_pca)
RMSa_als = sqrt(mean(alpha_als.^2,'all'))
RMSa_pca = sqrt(mean(alpha_pca.^2,'all'))
idvar_als = mean(var(alpha_als,0,1)./var(assets(241:T,:)-repmat(Y(241:T,6),[1,N]),0,1))
idvar_pca = mean(var(alpha_pca,0,1)./var(assets(241:T,:)-repmat(Y(241:T,6),[1,N]),0,1))