% tensor decomposition via PCA, Chen & Zimmerman factor zoo
% no regularization
clear;
rng(23,'twister');
Data1 = readtable('CZ01-20.csv','ReadVariableNames',true);
Date1 = unique(table2array(Data1(:,3)));
N = table2array(Data1(1,12));
J = 10;
T = 240;
rets = table2array(Data1(:,4));
Data2 = readtable('F-F_Research_Data_5_Factors_2x3.csv','ReadVariableNames',true,'VariableNamingRule','preserve');
Date2 = datetime(Data2.Date*100+1,'ConvertFrom','yyyymmdd','Format','yyyy-MM-dd');
startyear = 2001;
endyear = 2020;
first2 = find(Date2>=datetime(startyear,01,01),1);
last2 = find(Date2>=datetime(endyear,12,01),1);
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
% Mu
[Gamma_2,S_2] = eig(Y_2*Y_2');
% Gamma_2 = Gamma_2*sqrt(J);
[s_2,ind] = sort(diag(S_2),'descend');
M_hat = Gamma_2(:,ind(1:R));
% F
[Gamma_3,S_3] = eig(Y_3*Y_3');
% Gamma_3 = Gamma_3*sqrt(T);
[s_3,ind] = sort(diag(S_3),'descend');
L_hat = Gamma_3(:,ind(1:R));


weights = M_hat-mean(M_hat);
if mean(weights(1:5))-mean(weights(6:10))>0
    weights = weights*(-1); 
end
% 
% % Mean, Variance, Correlation comparison - CZ
% RP_ts = NaN(T,N);
% for n=1:N
%     RP_ts(:,n) = Y(:,:,n) * weights;
% end
% hml = [-1;zeros(8,1);1];
% RP_hml = NaN(T,N);
% for n=1:N
%     RP_hml(:,n) = Y(:,:,n) * hml;
% end
% std_ts = NaN(N,1);std_hml = NaN(N,1);
% mean_ts = NaN(N,1);mean_hml = NaN(N,1);
% for n=1:N
%     std_ts(n) = std(RP_ts(:,n));
%     std_hml(n) = std(RP_hml(:,n));
%     mean_ts(n) = mean(RP_ts(:,n));
%     mean_hml(n) = mean(RP_hml(:,n));
% end
% diff_std = std_hml - std_ts;
% diff_mean = mean_hml - mean_ts;
% ratio_std = std_hml./std_ts;
% ratio_mean = mean_hml./mean_ts;
% % r1(sample) = size(find(diff_std>0),1)/N;
% % r2(sample) = size(find(diff_mean<0),1)/N;
% % r3(sample) = mean(ratio_std);
% sr_ts = mean_ts./std_ts;
% sr_hml = mean_hml./std_hml;
% ratio_sr = sr_ts./sr_hml;
% for n=1:N
%     if mean_ts(n)<0 || mean_hml(n)<0
%     ratio_sr(n) = NaN;
%     diff_mean(n) = NaN;
%     ratio_std(n) = NaN;
%     end
% end
% % for n=1:N
% %     if diff_std(n)<0
% %         diff_mean(n) = NaN;
% %     end
% % end
% [~,I1] = sort(ratio_sr,'descend','MissingPlacement','last');
% [~,I2] = sort(diff_mean,'ascend','MissingPlacement','last');
% [~,I3] = sort(ratio_std,'descend','MissingPlacement','last');
% I=I3;
% 
% for i=1:10
%     top{i,1} = char(Data1{(I(i)-1)*3000+1,5});
%     std_ts_top(i,1) = std_ts(I(i));
%     std_hml_top(i,1) = std_hml(I(i));
%     mean_ts_top(i,1) = mean_ts(I(i));
%     mean_hml_top(i,1) = mean_hml(I(i));
%     sr_ts_top(i,1) = sr_ts(I(i));
%     sr_hml_top(i,1) = sr_hml(I(i));
%     ratio_top(i,1) = ratio_std(I(i));
% end

% bootstrap
bs_n = 5000;
Y_bs = NaN(T,J,N,bs_n);
hml = [-1;zeros(8,1);1];
for n=1:N
    for j=1:J
        Y_bs(:,j,n,:) = stationary_bootstrap(Y(:,j,n),bs_n,6.21); %6.69=300^(1/3)
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
sr_ts = (mean_ts-mean(RF))./std_ts;
sr_hml = (mean_hml-mean(RF))./std_hml;
diff_mean = mean_ts-mean_hml;
ratio_std = std_ts./std_hml;
r1 = size(find(std_ts./std_hml<1),1)/N;
r2 = size(find(mean_ts-mean_hml>0),1)/N;
for n=1:N
    if mean_ts(n)<0 || mean_hml(n)<0 %|| diff_mean(n)<0
    diff_mean(n) = NaN;
    ratio_std(n) = NaN;
    end
end
r3 = mean(ratio_std,'omitnan');
[~,I1] = sort(diff_mean,'descend','MissingPlacement','last');
[~,I2] = sort(ratio_std,'ascend','MissingPlacement','last');
I = I2;
for i=1:10
    top{i,1} = char(Data1{(I(i)-1)*2400+1,5});
    std_ts_top(i,1) = std_ts(I(i));
    std_hml_top(i,1) = std_hml(I(i));
    mean_ts_top(i,1) = mean_ts(I(i));
    mean_hml_top(i,1) = mean_hml(I(i));
%     sr_ts_top(i,1) = sr_ts(I(i));
%     sr_hml_top(i,1) = sr_hml(I(i));
    ratio_top(i,1) = ratio_std(I(i));
end


top = cell2table(top,"VariableNames",{'2001-2020'});

stats = [mean_ts_top,std_ts_top,ratio_top];
stats = round(stats,3);
stats = array2table(stats,"VariableNames",{'RP','SD','ratio'});
tab = [top,stats];

scatter(sr_hml,sr_ts);
hold on
sl = (min([sr_hml,sr_ts],[],'all')-0.1):0.1:(max([sr_hml,sr_ts],[],'all')+0.1);
plot(sl,sl);
xlabel('SR\_HML','FontSize',12)
ylabel('SR\_TS','FontSize',12)
title('2001-2020','FontSize',14)
% exportgraphics(gcf,'sr0120.pdf','BackgroundColor','none','ContentType','vector')

% test SR_ts > SR_hml
diff_sr = sr_ts - sr_hml;
[h,p] = ttest(diff_sr,[],'Tail','right');

%% significance test
% se_ts=NaN(N);coef_ts=NaN(N);
% se_hml=NaN(N);coef_hml=NaN(N);
% for n=1:N
%     [~,se_ts(n),coef_ts(n)] = hac(ones(300,1),RP_ts(:,n),'Intercept',false,Display="off");
%     [~,se_hml(n),coef_hml(n)] = hac(ones(300,1),RP_hml(:,n),'Intercept',false,Display="off");
%     p_ts(n) = 2*(1-normcdf(abs(coef_ts(n)/se_ts(n))));
%     p_hml(n) = 2*(1-normcdf(abs(coef_hml(n)/se_hml(n))));
% end
% 
% sig_ts = find(p_ts<0.05);
% sig_hml = find(p_hml<0.05);
% 
% C=intersect(sig_ts,sig_hml);
% sig_ts_hml=setdiff(sig_ts,C);
% sig_hml_ts=setdiff(sig_hml,C);
% 
% for i=1:size(sig_ts_hml,2)
%     sig(1,i) = Data1{(sig_ts_hml-1)*3000+1,5}(i);
% end
% for i=1:size(sig_hml_ts,2)
%     sig(2,i) = Data1{(sig_hml_ts-1)*3000+1,5}(i);
% end
% 
% table2latex(cell2table(sig))