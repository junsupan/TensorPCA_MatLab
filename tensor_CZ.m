% tensor decomposition via PCA, Chen & Zimmerman factor zoo
% no regularization
clear;
top = cell(10,5);
sig_ts = {};sig_ts_hml = {};
sig_hml = {};sig_hml_ts = {};
sig = {};
for sample = 1:5
range = {'96-00','01-05','06-10','11-15','16-20'};
Data1 = readtable(strcat('CZ',range{sample},'.csv'),'ReadVariableNames',true);
Date1 = unique(table2array(Data1(:,3)));
N = table2array(Data1(1,12));
Num(sample) = N;
J = 10;
T = 60;
rets = table2array(Data1(:,4));
Data2 = readtable('F-F_Research_Data_5_Factors_2x3.csv','ReadVariableNames',true,'VariableNamingRule','preserve');
Date2 = datetime(Data2.Date*100+1,'ConvertFrom','yyyymmdd','Format','yyyy-MM-dd');
startyear = [1996,2001,2006,2011,2016];
endyear = [2000,2005,2010,2015,2020];
first2 = find(Date2>=datetime(startyear(sample),01,01),1);
last2 = find(Date2>=datetime(endyear(sample),12,01),1);
mkt = table2array(Data2(first2:last2,"Mkt-RF"));% + table2array(Data2(first2:last2,"RF"));
HML = table2array(Data2(first2:last2,"HML"));
RMW = table2array(Data2(first2:last2,"RMW"));
CMA = table2array(Data2(first2:last2,"CMA"));

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
        [~,~,Y_mkt(:,j,n)] = regress(Y(:,j,n),mkt);
    end
end
% statistics of the data
mean_mean(sample,:) = mean(avg,1);
mean_stdev(sample,:) = std(avg,1);
stdev_mean(sample,:) = mean(stdev,1);
stdev_stdev(sample,:) = std(stdev,1);
%
Y_mkt = tensor(Y_mkt);
Y_1 = double(reshape(Y_mkt,[T,J*N]));
Y_2 = double(reshape(permute(Y_mkt,[2,1,3]),[J,T*N]));
Y_3 = double(reshape(permute(Y_mkt,[3,1,2]),[N,T*J]));

% tensor decomposition via PCA 
% A
R = 1; % rank
[Gamma_1,S_1] = eig(Y_1*Y_1');
A = NaN(T,R);
for r=1:R
    A(:,r) = Gamma_1(:,T-r+1);
end
s_1 = NaN(R,1);
for r=1:R
    s_1(r) = sqrt(S_1(T-r+1,T-r+1));
end
A_copy(:,sample) = A;
% B
[Gamma_2,S_2] = eig(Y_2*Y_2');
B = NaN(J,R);
for r=1:R
    B(:,r) = Gamma_2(:,J-r+1);
end
s_2 = NaN(R,1);
for r=1:R
    s_2(r) = sqrt(S_2(J-r+1,J-r+1));
end
B_copy(:,sample) = B;
% C
[Gamma_3,S_3] = eig(Y_3*Y_3');
C = NaN(N,R);
for r=1:R
    C(:,r) = Gamma_3(:,N-r+1);
end
s_3 = NaN(R,1);
for r=1:R
    s_3(r) = sqrt(S_3(N-r+1,N-r+1));
end
C_copy{sample} = C;

weights = B-mean(B);
weights_table(:,sample) = weights;
if mean(weights(1:5))-mean(weights(6:10))>0
    weights = weights*(-1);
end
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
RP_ts = NaN(T,N);
for n=1:N
    RP_ts(:,n) = Y(:,:,n) * weights;
end
hml = [-1;zeros(8,1);1];
RP_hml = NaN(T,N);
for n=1:N
    RP_hml(:,n) = Y(:,:,n) * hml;
end
stdev_ts = NaN(N,1);stdev_hml = NaN(N,1);
mean_ts = NaN(N,1);mean_hml = NaN(N,1);
for n=1:N
    stdev_ts(n) = std(RP_ts(:,n));
    stdev_hml(n) = std(RP_hml(:,n));
    mean_ts(n) = mean(RP_ts(:,n));
    mean_hml(n) = mean(RP_hml(:,n));
end
diff_std = stdev_hml - stdev_ts;
diff_mean = mean_hml - mean_ts;
ratio_std = stdev_hml./stdev_ts;
ratio_mean = mean_hml./mean_ts;
r1(sample) = size(find(diff_std>0),1)/N;
r2(sample) = size(find(diff_mean<0),1)/N;
r3(sample) = mean(ratio_std);
sr_ts = mean_ts./stdev_ts;
sr_hml = mean_hml./stdev_hml;
ratio_sr = sr_ts./sr_hml;
for n=1:N
    if sr_ts(n)<0 || sr_hml(n)<0
    ratio_sr(n) = 0;
    diff_mean(n) = 100;
    ratio_std(n) = 0;
    end
end
for n=1:N
    if diff_std(n)<0
        diff_mean(n) = 100;
    end
end
[~,I1] = sort(ratio_sr,'descend');
[~,I2] = sort(diff_mean,'ascend');
[~,I3] = sort(ratio_std,'descend');
I=I3;

for i=1:10
    top{i,sample} = char(Data1{(I(i)-1)*600+1,5});
    stdev_ts_top(i,sample) = stdev_ts(I(i));
    stdev_hml_top(i,sample) = stdev_hml(I(i));
    mean_ts_top(i,sample) = mean_ts(I(i));
    mean_hml_top(i,sample) = mean_hml(I(i));
    sr_ts_top(i,sample) = sr_ts(I(i));
    sr_hml_top(i,sample) = sr_hml(I(i));
    ratio_top(i,sample) = ratio_std(I(i));
end

% significance test
% se_ts=NaN(N);coef_ts=NaN(N);
% se_hml=NaN(N);coef_hml=NaN(N);
% for n=1:N
%     [~,se_ts(n),coef_ts(n)] = hac(ones(60,1),RP_ts(:,n),'Intercept',false,Display="off");
%     [~,se_hml(n),coef_hml(n)] = hac(ones(60,1),RP_hml(:,n),'Intercept',false,Display="off");
%     p_ts(n) = 2*(1-normcdf(abs(coef_ts(n)/se_ts(n))));
%     p_hml(n) = 2*(1-normcdf(abs(coef_hml(n)/se_hml(n))));
% end
% 
% sig_ts(sample) = {find(p_ts<0.05)};
% sig_hml(sample) = {find(p_hml<0.05)};
% 
% C=intersect(sig_ts{sample},sig_hml{sample});
% sig_ts_hml{sample}=setdiff(sig_ts{sample},C);
% sig_hml_ts{sample}=setdiff(sig_hml{sample},C);
% 
% for i=1:size(sig_ts_hml{sample},2)
%     sig(2*(sample-1)+1,i) = Data1{(sig_ts_hml{sample}-1)*600+1,5}(i);
% end
% for i=1:size(sig_hml_ts{sample},2)
%     sig(2*sample,i) = Data1{(sig_hml_ts{sample}-1)*600+1,5}(i);
% end


end

%% tables
[uniqueXX, ~, c]=unique(top);
occ = histc(c, 1:numel(uniqueXX));
% uniqueXX(find(occ==4))
% nnz(strcmp(top20,'Cash'))
top = cell2table(top,"VariableNames",{'1996-2000','2001-2005','2006-2010','2011-2015','2016-2020'});


% table1 = [top(:,1),array2table(round([mean_ts_top(:,1),stdev_ts_top(:,1),sr_ts_top(:,1),...
%     mean_hml_top(:,1),stdev_hml_top(:,1),sr_hml_top(:,1)],2),"VariableNames",...
%     {'RP\_ts','SD\_ts','SR\_ts','RP\_hml','SD\_hml','SR\_hml'})];
% table2 = [top(:,2),array2table(round([mean_ts_top(:,2),stdev_ts_top(:,2),sr_ts_top(:,2),...
%     mean_hml_top(:,2),stdev_hml_top(:,2),sr_hml_top(:,2)],2),"VariableNames",...
%     {'RP\_ts','SD\_ts','SR\_ts','RP\_hml','SD\_hml','SR\_hml'})];
% table3 = [top(:,3),array2table(round([mean_ts_top(:,3),stdev_ts_top(:,3),sr_ts_top(:,3),...
%     mean_hml_top(:,3),stdev_hml_top(:,3),sr_hml_top(:,3)],2),"VariableNames",...
%     {'RP\_ts','SD\_ts','SR\_ts','RP\_hml','SD\_hml','SR\_hml'})];
% table4 = [top(:,4),array2table(round([mean_ts_top(:,4),stdev_ts_top(:,4),sr_ts_top(:,4),...
%     mean_hml_top(:,4),stdev_hml_top(:,4),sr_hml_top(:,4)],2),"VariableNames",...
%     {'RP\_ts','SD\_ts','SR\_ts','RP\_hml','SD\_hml','SR\_hml'})];
% table5 = [top(:,5),array2table(round([mean_ts_top(:,5),stdev_ts_top(:,5),sr_ts_top(:,5),...
%     mean_hml_top(:,5),stdev_hml_top(:,5),sr_hml_top(:,5)],2),"VariableNames",...
%     {'RP\_ts','SD\_ts','SR\_ts','RP\_hml','SD\_hml','SR\_hml'})];

stats = [];
for i=1:5
    stats = [stats,mean_ts_top(:,i),stdev_ts_top(:,i),ratio_top(:,i)];
end
stats = array2table(stats,"VariableNames",{'RP1','SE1','ratio1','RP2','SE2','ratio2',...
    'RP3','SE3','ratio3','RP4','SE4','ratio4','RP5','SE5','ratio5'})
tab = [top(:,1),stats(:,1:3)];
for i=2:5
    tab = [tab,top(:,i),stats(:,3*(i-1)+1:3*i)];
end


table2latex(tab(:,1:8))
table2latex(tab(:,9:16))
table2latex([tab(:,17:20),cell2table(cell(10,4))])


%% significance test
sig = NaN(10,22);
for i = 1:5
    sig(2*(i-1)+1,1:size(sig_ts{i},2)) = sig_ts{i};
    sig(2*i,1:size(sig_hml{i},2)) = sig_hml{i};
end
sig = array2table(sig,"RowNames",{'1_ts','1_hml','2_ts','2_hml','3_ts','3_hml','4_ts','4_hml','5_ts','5_hml'});
% table2latex(sig)
for i=1:5
    C=intersect(sig_ts{i},sig_hml{i});
    sig_ts_hml{i}=setdiff(sig_ts{i},C);
    sig_hml_ts{i}=setdiff(sig_hml{i},C);
end

%%
stats = [];
for i=1:5
    stats = [stats;mean_mean(i,:);mean_stdev(i,:);stdev_mean(i,:);stdev_stdev(i,:)];
end
results = [];
for i=1:5
    results = [results;[r1(i),r3(i),r2(i)]];
end
results = array2table(round(results,2),"VariableNames",{'Variance','avg. var. ratio','Mean'});
results = [cell2table({'1996-2000';'2001-2005';'2006-2010';'2011-2015';'2016-2020'},"VariableNames",{'Year'}),results];

rownames = {};
for i=1:5
rownames = [rownames,{'mean of mean','s.d. of mean','mean of s.d.','s.d. of s.d.'}];
end
varnames = sprintfc('%d',1:10);
% table2latex(array2table(round(stats,2),"VariableNames",varnames))
% table2latex(results)


% plot(RP_hml(:,10))
% hold on
% plot(RP_ts(:,10))

%% Mean, Variance, Correlation comparison - FF
Data3 = readtable('Portfolios_Formed_on_ME-2.csv','ReadVariableNames',true,'VariableNamingRule','preserve');
% Data3 = readtable('Portfolios_Formed_on_BE-ME.csv','ReadVariableNames',true,'VariableNamingRule','preserve');
Date3 = datetime(Data3.Date*100+1,'ConvertFrom','yyyymmdd','Format','yyyy-MM-dd');
startyear = [1996,2001,2006,2011,2016];
endyear = [2000,2005,2010,2015,2020];
first3 = find(Date3>=datetime(startyear(sample),01,01),1);
last3 = find(Date3>=datetime(endyear(sample),12,01),1);
ME = table2array(Data3(first3:last3,11:20));
ME = ME(:,J:-1:1); % comment for BE/ME
RP = ME * weights;
RP_long((i-1)*T+1:i*T,1) = RP;
% size_new_org = ME * weights_org;
RP_FF = table2array(Data2(first2:last2,"SMB")); % HML for BE/ME
RP_FF_long((i-1)*T+1:i*T,1) = RP_FF;
stats(sample,:) = round([mean(RP),var(RP),mean(RP_FF),var(RP_FF),corr(RP,RP_FF)],4);

%%
varnames = {'mean_new','var_new','mean_FF','var_FF','corr'};
stats = array2table(stats,'VariableNames',varnames);

first = find(Date3>=datetime(1996,01,01),1);
last = find(Date3>=datetime(2020,12,01),1);
Date = Date3(first:last);
% RP_long = RP_long/std(RP_long);
% RP_FF_long = RP_FF_long/std(RP_FF_long);
figure;
plot(Date,RP_long);
hold on;
% plot(Date1,size_new_org);
plot(Date,RP_FF_long);
recessionplot
legend('CP premium','FF SMB');
title('risk premia');
set(gcf, 'Position',  [500, 500, 1200, 400])

diff = RP_long - RP_FF_long;
figure;
plot(Date,diff);
recessionplot
legend('difference');
title('difference of two risk premia');
set(gcf, 'Position',  [500, 500, 1200, 400])

%% tests
[~,se1,coef1] = hac(ones(300,1),diff,'Intercept',false);
coef1
se1
p1 = 2*(1-normcdf(abs(coef1/se1)))
for i=1:5
    [~,se_sub(i),coef_sub(i)] = hac(ones(60,1),diff(1+(i-1)*60:i*60),'Intercept',false);
    p_sub(i) = 2*(1-normcdf(abs(coef_sub(i)/se_sub(i))));
end
% mdl = fitlm(ones(300,1),diff,'Intercept',false);
rec1b = find(Date>=datetime(2001,03,01),1);
rec1e = find(Date>=datetime(2001,11,01),1);
rec2b = find(Date>=datetime(2007,12,01),1);
rec2e = find(Date>=datetime(2009,06,01),1);
recession = zeros(300,1);
for t=1:300
    if (t>=rec1b && t<=rec1e) || (t>=rec2b && t<=rec2e)
        recession(t) = 1;
    end
end
[~,se2,coef2] = hac([ones(300,1),recession],diff,'Intercept',false);
coef2
se2
p2 = 2*(1-normcdf(abs(coef2(2)/se2(2))))
Mdl = arima(1,0,0);
EstMdl = estimate(Mdl,diff);
EstMdl2 = estimate(Mdl,diff.^2);

%% Fama-MacBeth
first2 = find(Date2>=datetime(1996,01,01),1);
last2 = find(Date2>=datetime(2020,12,01),1);
mkt = table2array(Data2(first2:last2,"Mkt-RF"));% + table2array(Data2(first2:last2,"RF"));
HML = table2array(Data2(first2:last2,"HML"));
RMW = table2array(Data2(first2:last2,"RMW"));
CMA = table2array(Data2(first2:last2,"CMA"));

Data4 = readtable('CZ96-20.csv','ReadVariableNames',true);
Date4 = unique(table2array(Data4(:,3)));
N4 = table2array(Data4(1,12));
T4 = 300;
J=10;
rets = table2array(Data4(:,4));
assets = NaN(T4,J,N4);
for n=1:N4
    assets(:,:,n) = reshape(rets((n-1)*T4*J+1:n*T4*J),T4,J);
end
avg=NaN(N4,J);
for n=1:N4
    avg(n,:) = mean(assets(:,:,n),1);
end
stdev=NaN(N4,J);
for n=1:N4
    stdev(n,:) = std(assets(:,:,n),1);
end
for n=1:N4
    if mean(avg(n,1:5))-mean(avg(n,6:10))>0
        avg(n,:) = avg(n,J:-1:1);
    end
end
mean(avg,1)
std(avg,1)
mean(stdev,1)

% distribution of estimations, for simulation calibration
histfit(A_copy(:,3))
fitdist(A_copy(:,3),'Normal')
chi2gof(A_copy(:,3))
histfit(C_copy{3}-min(C_copy{3})+0.001,10,'Gamma')
pd=fitdist(C_copy{3}-min(C_copy{3})+0.001,'gamma')
chi2gof(C_copy{3}-min(C_copy{3})+0.001,'cdf',pd)

