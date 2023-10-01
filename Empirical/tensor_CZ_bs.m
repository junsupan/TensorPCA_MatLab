% tensor decomposition via PCA, Chen & Zimmerman factor zoo
% no regularization
clear;
top = cell(10,4);
rng(23,'twister');
for sample = 1:4
range = {'01-05','06-10','11-15','16-20'};
Data1 = readtable(strcat('CZ',range{sample},'.csv'),'ReadVariableNames',true);
Date1 = unique(table2array(Data1(:,3)));
N = table2array(Data1(1,12));
Num(sample) = N;
J = 10;
T = 60;
rets = table2array(Data1(:,4));
Data2 = readtable('F-F_Research_Data_5_Factors_2x3.csv','ReadVariableNames',true,'VariableNamingRule','preserve');
Date2 = datetime(Data2.Date*100+1,'ConvertFrom','yyyymmdd','Format','yyyy-MM-dd');
startyear = [2001,2006,2011,2016];
endyear = [2005,2010,2015,2020];
first2 = find(Date2>=datetime(startyear(sample),01,01),1);
last2 = find(Date2>=datetime(endyear(sample),12,01),1);
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
Y_mkt = NaN(T,J,N);
for j=1:J
    for n=1:N
        [~,~,Y_mkt(:,j,n)] = regress(Y(:,j,n)-RF,mkt);
    end
end
% statistics of the data
% mean_mean(sample,:) = mean(avg,1);
% mean_stdev(sample,:) = std(avg,1);
% stdev_mean(sample,:) = mean(stdev,1);
% stdev_stdev(sample,:) = std(stdev,1);
%
Y_mkt = tensor(Y_mkt);
Y_1 = double(reshape(Y_mkt,[T,J*N]));
Y_2 = double(reshape(permute(Y_mkt,[2,1,3]),[J,T*N]));
Y_3 = double(reshape(permute(Y_mkt,[3,1,2]),[N,T*J]));


% tensor decomposition via PCA 
R=1;
% Lambda
[Gamma_1,S_1] = eig(Y_1*Y_1');
% Gamma_1 = Gamma_1*sqrt(N);
[s_1,ind] = sort(diag(S_1),'descend');
F_hat = Gamma_1(:,ind(1:R));
F_copy(:,sample) = F_hat;
% Mu
[Gamma_2,S_2] = eig(Y_2*Y_2');
% Gamma_2 = Gamma_2*sqrt(J);
[s_2,ind] = sort(diag(S_2),'descend');
M_hat = Gamma_2(:,ind(1:R));
M_copy(:,sample) = M_hat;
% F
[Gamma_3,S_3] = eig(Y_3*Y_3');
% Gamma_3 = Gamma_3*sqrt(T);
[s_3,ind] = sort(diag(S_3),'descend');
L_hat = Gamma_3(:,ind(1:R));
L_copy{sample} = L_hat;


weights = M_hat-mean(M_hat);
if mean(weights(1:5))-mean(weights(6:10))>0
    weights = weights*(-1);
end
weights_table(:,sample) = weights;
% s_copy(sample,:) = [s_1,s_2,s_3];
% ALS
% CPALS = cp_als(Y_mkt,R,'printitn',0);
% s_als(sample) = CPALS.lambda;
% Y_hat = ktensor(CPALS.lambda,{A,B,C});
% Y_hat = tensor(Y_hat);
% u = Y_mkt-Y_hat;
% u_1 = double(reshape(u,[T,J*N]));
% u = double(u)
% chi2gof(u(:,1,1))

% Without market taken out
% Y = tensor(Y);
% Y_2_org = double(reshape(permute(Y,[2,1,3]),[J,T*N]));
% [Gamma_2,S_2] = eig(Y_2_org*Y_2_org');
% B_org = NaN(J,R);
% for r=1:R
%     B_org(:,r) = Gamma_2(:,J-r+1);
% end
% weights_org = B_org-mean(B_org);
% weights_org_table(:,sample) = weights_org;

% Mean, Variance, Correlation comparison - CZ
% RP_ts = NaN(T,N);
% for n=1:N
%     RP_ts(:,n) = Y(:,:,n) * weights;
% end
% hml = [-1;zeros(8,1);1];
% RP_hml = NaN(T,N);
% for n=1:N
%     RP_hml(:,n) = Y(:,:,n) * hml;
% end
% stdev_ts = NaN(N,1);stdev_hml = NaN(N,1);
% mean_ts = NaN(N,1);mean_hml = NaN(N,1);
% for n=1:N
%     stdev_ts(n) = std(RP_ts(:,n));
%     stdev_hml(n) = std(RP_hml(:,n));
%     mean_ts(n) = mean(RP_ts(:,n));
%     mean_hml(n) = mean(RP_hml(:,n));
% end
% diff_std = stdev_hml - stdev_ts;
% diff_mean = mean_hml - mean_ts;
% ratio_std = stdev_hml./stdev_ts;
% ratio_mean = mean_hml./mean_ts;
% r1(sample) = size(find(diff_std>0),1)/N;
% r2(sample) = size(find(diff_mean<0),1)/N;
% r3(sample) = mean(ratio_std);
% sr_ts = mean_ts./stdev_ts;
% sr_hml = mean_hml./stdev_hml;
% ratio_sr = sr_ts./sr_hml;
% for n=1:N
%     if sr_ts(n)<0 || sr_hml(n)<0
%     ratio_sr(n) = 0;
%     diff_mean(n) = 100;
%     ratio_std(n) = 0;
%     end
% end
% for n=1:N
%     if diff_std(n)<0
%         diff_mean(n) = 100;
%     end
% end
% [~,I1] = sort(ratio_sr,'descend');
% [~,I2] = sort(diff_mean,'ascend');
% [~,I3] = sort(ratio_std,'descend');
% I=I3;
% 
% for i=1:10
%     top{i,sample} = char(Data1{(I(i)-1)*600+1,5});
%     stdev_ts_top(i,sample) = stdev_ts(I(i));
%     stdev_hml_top(i,sample) = stdev_hml(I(i));
%     mean_ts_top(i,sample) = mean_ts(I(i));
%     mean_hml_top(i,sample) = mean_hml(I(i));
%     sr_ts_top(i,sample) = sr_ts(I(i));
%     sr_hml_top(i,sample) = sr_hml(I(i));
%     ratio_top(i,sample) = ratio_std(I(i));
% end

% bootstrap
bs_n = 5000;
Y_bs = NaN(T,J,N,bs_n);
hml = [-1;zeros(8,1);1];
for n=1:N
    for j=1:J
        Y_bs(:,j,n,:) = stationary_bootstrap(Y(:,j,n),bs_n,3.92); %3.92=60^(1/3)
    end
end
mean_ts_bs = NaN(N,bs_n);mean_hml_bs = NaN(N,bs_n);
for bs=1:bs_n
    RP_ts_bs = NaN(T,N);
    for n=1:N
        RP_ts_bs(:,n) = Y_bs(:,:,n,bs) * weights;
    end
    RP_hml_bs = NaN(T,N);
    for n=1:N
        RP_hml_bs(:,n) = Y_bs(:,:,n,bs) * hml;
    end
    for n=1:N
        mean_ts_bs(n,bs) = mean(RP_ts_bs(:,n));
        mean_hml_bs(n,bs) = mean(RP_hml_bs(:,n));
    end
end

std_ts = std(mean_ts_bs,0,2);
std_hml = std(mean_hml_bs,0,2);
mean_ts = mean(mean_ts_bs,2);
mean_hml = mean(mean_hml_bs,2);
sr_ts{sample} = (mean_ts-mean(RF))./std_ts;
sr_hml{sample} = (mean_hml-mean(RF))./std_hml;
diff_mean = mean_ts-mean_hml;
ratio_std = std_ts./std_hml;
r1(sample) = size(find(std_ts./std_hml<1),1)/N;
r2(sample) = size(find(mean_ts-mean_hml>0),1)/N;

for n=1:N
    if mean_ts(n)<0 || mean_hml(n)<0 %|| diff_mean(n)<0
    diff_mean(n) = NaN;
    ratio_std(n) = NaN;
    end
end
% for n=1:N
%     if sr_ts(n,sample)<0
%         sr_ts(n,sample) = NaN;
%     end
%     if sr_hml(n,sample)<0
%         sr_hml(n,sample) = NaN;
%     end
% end
        
r3(sample) = mean(ratio_std,'omitnan');
[~,I1] = sort(diff_mean,'descend','MissingPlacement','last');
[~,I2] = sort(ratio_std,'ascend','MissingPlacement','last');
I = I2;
for i=1:10
    top{i,sample} = char(Data1{(I(i)-1)*600+1,5});
    std_ts_top(i,sample) = std_ts(I(i));
    std_hml_top(i,sample) = std_hml(I(i));
    mean_ts_top(i,sample) = mean_ts(I(i));
    mean_hml_top(i,sample) = mean_hml(I(i));
%     sr_ts_top(i,sample) = sr_ts(I(i));
%     sr_hml_top(i,sample) = sr_hml(I(i));
    ratio_top(i,sample) = ratio_std(I(i));
end

end

top = cell2table(top,"VariableNames",{'2001-2005','2006-2010','2011-2015','2016-2020'});

stats = [];
for i=1:4
    stats = [stats,mean_ts_top(:,i),std_ts_top(:,i),ratio_top(:,i)];
end
stats = round(stats,3);
stats = array2table(stats,"VariableNames",{'RP1','SD1','ratio1','RP2','SD2','ratio2',...
    'RP3','SD3','ratio3','RP4','SD4','ratio4'});
tab = [top(:,1),stats(:,1:3)];
for i=2:4
    tab = [tab,top(:,i),stats(:,3*(i-1)+1:3*i)];
end

% plot sharpe ratio
i=4;
scatter(sr_hml{i},sr_ts{i});
hold on
sl = (min([sr_hml{i},sr_ts{i}],[],'all')-0.1):0.1:(max([sr_hml{i},sr_ts{i}],[],'all')+0.1);
plot(sl,sl);
xlabel('SR\_HML','FontSize',12)
ylabel('SR\_TS','FontSize',12)
title('2016-2020','FontSize',14)
% exportgraphics(gcf,'sr.pdf','BackgroundColor','none','ContentType','vector')

% test SR_ts > SR_hml
for i=1:4
    diff_sr{i} = sr_ts{i} - sr_hml{i};
    [h(i),p(i)] = ttest(diff_sr{i},[],'Tail','right');
end
