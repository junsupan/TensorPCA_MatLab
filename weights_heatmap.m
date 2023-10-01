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
end
%%
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

%%
hml = [-1;zeros(8,1);1];
yvalues = strcat('decile',{' '},cellstr(string(1:10)));
year = {'2001-2005','2006-2010','2011-2015','2016-2020','2001-2020','high-minus-low'};
heatmap(year,yvalues,round([weights_table,weights,hml],3));
exportgraphics(gcf,'weights.pdf','BackgroundColor','none','ContentType','vector')