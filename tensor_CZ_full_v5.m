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
% recover factor strength
s0 = (s1+s2+s3)/3;
[s_ts,~] = findsigma2(Y,{F_hat,M_hat,L_hat},s0);

% root mean squared error, idiosyncratic variation
n = min([N*J,N*T,J*T]);
rmse_ts = nan(R,1);R2_ts = nan(R,1);adjR2_ts = nan(R,1);AIC_ts = nan(R,1);
for r=1:R
    Y_e = Y - double(tensor(ktensor(s_ts(1:r),{F_hat(:,1:r),M_hat(:,1:r),L_hat(:,1:r)})));
    rmse_ts(r) = rms(Y_e,'all');
    R2_ts(r) = 1 - var(Y_e,0,'all')/var(Y,0,'all');
%     R2_ts(r) = 1 - sum(Y_e.^2,'all')/sum(Y.^2,'all');
    adjR2_ts(r) = 1 - (1-R2_ts(r))*(N*J*T-1)/(N*J*T-r*(N+J+T));
    AIC_ts(r) = sum(Y_e.^2,'all')/NN + r*(N+J+T)/n;
end

% step-wise ALS
% Yr = Y;
% F_als = nan(T,R);M_als = nan(J,R);L_als = nan(N,R);s_als = nan(R,1);
% rmse_als = nan(R,1);R2_als = nan(R,1);adjR2_als = nan(R,1);AIC_als = nan(R,1);
% for r=1:R
%     M = cp_als(tensor(Yr),1,'printitn',0);
%     F_als(:,r) = M.U{1};
%     M_als(:,r) = M.U{2};
%     L_als(:,r) = M.U{3};
%     s_als(r) = M.lambda;
%     Yr = Yr - double(tensor(ktensor(s_als(r),{F_als(:,r),M_als(:,r),L_als(:,r)})));
%     rmse_als(r) = rms(Yr,'all');
%     R2_als(r) = 1 - var(Yr,0,'all')/var(Y,0,'all');
% %     R2_als(r) = 1 - sum(Yr.^2,'all')/sum(Y.^2,'all');
%     adjR2_als(r) = 1 - (1-R2_als(r))*(N*J*T-1)/(N*J*T-r*(N+J+T));
%     AIC_als(r) = sum(Yr.^2,'all')/NN + r*(N+J+T)*log(NN)/NN;
% end

R=3;
M = cp_als(tensor(Y),R,'printitn',0);
F_als = M.U{1};
M_als = M.U{2};
L_als = M.U{3};
s_als = M.lambda;
rmse_als = nan(R,1);R2_als = nan(R,1);
for r=1:R
    Y_e = Y - double(tensor(ktensor(s_als(1:r),{F_als(:,1:r),M_als(:,1:r),L_als(:,1:r)})));
    rmse_als(r) = rms(Y_e,'all');
%     R2_als(r) = 1 - var(Y_e,0,'all')/var(Y,0,'all');
    R2_als(r) = 1 - sum(Y_e.^2,'all')/sum(Y.^2,'all');
end

R=5;
% SVD
[F_tw,S,Beta_tw] = svd(Y_1);
F_tw = F_tw(:,1:R);
Beta_tw = Beta_tw(:,1:R);
s_tw = diag(S);s_tw = s_tw(1:R);
rmse_tw = nan(R,1);R2_tw = nan(R,1);adjR2_tw = nan(R,1);AIC_tw = nan(R,1);
for r=1:R
    Y_1_e = Y_1 - F_tw(:,1:r)*diag(s_tw(1:r))*Beta_tw(:,1:r)';
    rmse_tw(r) = rms(Y_1_e,'all');
    R2_tw(r) = 1 - var(Y_1_e,0,'all')/var(Y,0,'all');
%     R2_tw(r) = 1 - sum(Y_1_e.^2,'all')/sum(Y.^2,'all');
    adjR2_tw(r) = 1 - (1-R2_tw(r))*(N*J*T-1)/(N*J*T-r*(N*J+T));
    AIC_tw(r) = sum(Y_1_e.^2,'all')/NN + r*(N*J+T)/n;
end
% Model complexity
cplxity_ts = NaN(1,R);cplxity_tw = NaN(1,R);
for r=1:R
    cplxity_ts(r) = (N+J+T)*r/(N*J*T);
    cplxity_tw(r) = (N*J+T)*r/(N*J*T);
end

%% ALS with orthogonalization
R=3;
if R>1
    for r=2:R
        for s=1:r-1
            L_als(:,r) = L_als(:,r)- L_als(:,r)'*L_als(:,s)*L_als(:,s)/(L_als(:,s)'*L_als(:,s));
        end
    end
end
L_als = L_als/(sqrtm(L_als'*L_als));

if R>1
    for r=2:R
        for s=1:r-1
            M_als(:,r) = M_als(:,r)- M_als(:,r)'*M_als(:,s)*M_als(:,s)/(M_als(:,s)'*M_als(:,s));
        end
    end
end
M_als = M_als/(sqrtm(M_als'*M_als));

if R>1
    for r=2:R
        for s=1:r-1
            F_als(:,r) = F_als(:,r)- F_als(:,r)'*F_als(:,s)*F_als(:,s)/(F_als(:,s)'*F_als(:,s));
        end
    end
end
F_als = F_als/(sqrtm(F_als'*F_als));

L_hat = L_als;

%% loadings
max(L_hat,[],1)
mean(L_hat,1)
min(L_hat,[],1)
std(L_hat,0,1)
size(find(L_hat(:,1)>0),1)/N
size(find(L_hat(:,2)>0),1)/N
size(find(L_hat(:,3)>0),1)/N
% top loadings
[~,ind] = sort(L_hat(:,1),'descend');
top1 = cell(10,1);
for i=1:10
    top1{i,1} = char(Data1{(ind(i)-1)*3600+1,'signalname'});
end
[~,ind] = sort(L_hat(:,1),'ascend');
bot1 = cell(10,1);
for i=1:10
    bot1{i,1} = char(Data1{(ind(i)-1)*3600+1,'signalname'});
end
[~,ind] = sort(abs(L_hat(:,2)),'descend');
top2 = cell(10,1);
for i=1:10
    top2{i,1} = char(Data1{(ind(i)-1)*3600+1,'signalname'});
end
[~,ind] = sort(abs(L_hat(:,2)),'ascend');
bot2 = cell(10,1);
for i=1:10
    bot2{i,1} = char(Data1{(ind(i)-1)*3600+1,'signalname'});
end
table = cell2table([top1,bot1,top2,bot2],"VariableNames",{'top10','bot10','top102','bot102'});
% table2latex(table);

Data3 = readtable('SignalDoc.csv','ReadVariableNames',true,'VariableNamingRule','preserve');
tab = {};
acronyms = sort(unique([top1;bot1;top2;bot2]));
for i=1:size(acronyms,1)
    ind = find(strcmp(Data3{:,"Acronym"},acronyms{i}));
    tab = [tab;Data3{ind,["LongDescription","Authors"]}];
end
tab = [acronyms,tab];
tab = cell2table(tab,"VariableNames",{'Acronym','Description','Authors'});

%% not used below
figure;
plot(cplxity_ts,R2_ts,'-s','LineWidth',2, ...
    'MarkerSize',10,...
    'Color',[0 0.4470 0.7410],...
    'MarkerFaceColor',[0 0.4470 0.7410],...
    'MarkerEdgeColor','k');
hold on;
plot(cplxity_tw,R2_tw,'-s','LineWidth',2, ...
    'MarkerSize',10,...
    'Color',[0.8500 0.3250 0.0980],...
    'MarkerFaceColor',[0.8500 0.3250 0.0980],...
    'MarkerEdgeColor','k');
ylabel('R^2');
xlabel('Model Complexity')
for r=1:R
    text(cplxity_ts(r),R2_ts(r)-0.02,num2str(r));
    text(cplxity_tw(r),R2_tw(r)-0.02,num2str(r));
end
grid on;
% yL = [0.2 1.02];
% yL = [0.4 1];
% ylim(yL);
% line([cplxity_tw(1), cplxity_tw(1)], yL, 'LineWidth', 1, 'Color', [0.4660 0.6740 0.1880],'LineStyle','--');
% line([cplxity_tw(2), cplxity_tw(2)], yL, 'LineWidth', 1, 'Color', [0.4660 0.6740 0.1880],'LineStyle','--');
legend('3-way','2-way','Location','southeast');
set(gcf, 'Position',  [500, 500, 600, 400]);
set(gca,'XMinorTick','on','YMinorTick','on');
% exportgraphics(gcf,'sim10factors.pdf','BackgroundColor','none','ContentType','vector')


%% construct risk premium
weights = M_hat(:,2)-mean(M_hat(:,2));
if mean(weights(1:5))-mean(weights(6:10))>0
    weights = weights*(-1); 
end

RP_ts = NaN(T,N);
for n=1:N
    RP_ts(:,n) = Y(:,:,n) * weights;
end
hml = [-1;zeros(8,1);1];
RP_hml = NaN(T,N);
for n=1:N
    RP_hml(:,n) = Y(:,:,n) * hml;
end
std_ts = std(RP_ts,0,1);
std_hml = std(RP_hml,0,1);
mean_ts = mean(RP_ts,1);
mean_hml = mean(RP_hml,1);
sr_ts = (mean_ts-mean(RF))./std_ts;
sr_hml = (mean_hml-mean(RF))./std_hml;

scatter(sr_hml,sr_ts);
hold on
sl = (min([sr_hml,sr_ts],[],'all')-0.1):0.1:(max([sr_hml,sr_ts],[],'all')+0.1);
plot(sl,sl);
xlabel('SR\_HML','FontSize',12)
ylabel('SR\_TS','FontSize',12)

% test SR_ts > SR_hml
diff_sr = sr_ts - sr_hml;
[h,p] = ttest(diff_sr,[],'Tail','right');
