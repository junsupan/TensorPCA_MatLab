% TW distributed test

clear;clc;
rng(23,'twister');
R0 = 2; % true rank
R = 1; %rank estimated
T = 100;
N = 80;
J = 60;
K = 50000; %# of repetitions

% calculate critical value
K0 = 5;% parameter in TW distribution
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
%     Z(1:1001:end) = sqrt(9/(5/3)^2-1)*randn(1000,1);% t-errors
% variance is calculated as gamma_4/sigma^4 - 1


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
%% plot the pdf
histogram(asydist,20000,'Normalization','pdf');
xlim([0,15]);
hold on;
% yL = get(gca,'YLim');
line([c,c], [0,0.35], 'LineWidth', 1, 'Color', [0.4660 0.6740 0.1880],'LineStyle','--');
a=[cellstr(num2str(get(gca,'ytick')'*100))];
pct = char(ones(size(a,1),1)*'%'); 
new_yticks = [char(a),pct];
set(gca,'yticklabel',new_yticks)
text(c,0.2,'Critical Value ','HorizontalAlignment','right','Color',[0.4660 0.6740 0.1880],'FontSize',14)
%ytickformat('percentage');
%%
% generate Lambda
L = get_orthonormal(N,R0);
% L_norm = L/(sqrtm(L'*L));

% generate Mu
M = get_orthonormal(J,R0);
% M_norm = M/(sqrtm(M'*M));

% generate F
rho = 0.5;
sig_e = 0.1;
F = NaN(T+100,R0);
e = randn(T+100,R0);
F(1,:) = sig_e.*e(1,:);
for t=2:(T+100)
    F(t,:) = F(t-1,:).*rho + sig_e .* e(t,:);
end
F = F(101:T+100,:);
F = F/(sqrtm(F'*F));
% F_norm = F/(sqrtm(F'*F));

% d2 = 0;
d2=0:0.01:0.15; %0.12 for normal, 0.15 for t
% d2 = 1;
% power = NaN(length(d2),3);
p1 = NaN(K,length(d2));p2 = NaN(K,length(d2));p3 = NaN(K,length(d2));
for i=1:length(d2)
    disp(d2(i));

% signal strength
s = sqrt(N*J*T)*([2,d2(i)])';D=diag(s);

Sk1 = NaN(K,1);Sk2 = NaN(K,1);Sk3 = NaN(K,1);
for k = 1:K
%     if mod(k, 100) == 0
%         disp(k);
%     end
    % generate tensor Y   https://www.tensortoolbox.org/ktensor_doc.html#7
    sig_u = 1;
    % s = [2*sqrt((N*J*T)^a1);*sqrt((N*J*T)^a2)];
    Y = ktensor(s,{L,M,F});
    Y = tensor(Y);
    U = sig_u*randn([N,J,T]);
%     U = sig_u*sqrt(3/5)*trnd(5,N,J,T); % t-dist errors
    Y = double(Y) + U;
    Y_1 = reshape(Y,[N,J*T]);
    Y_2 = reshape(permute(Y,[2,1,3]),[J,N*T]);
    Y_3 = reshape(permute(Y,[3,1,2]),[T,N*J]);

    % tensor decomposition via PCA
    % Lambda
    [Gamma,S] = eig(Y_1*Y_1');
    [s1,ind] = sort(diag(S),'descend');
    L_hat = Gamma(:,ind(1:R));
%     s1 = sqrt(s1(1:R));
    % Mu
    [Gamma,S] = eig(Y_2*Y_2');
    [s2,ind] = sort(diag(S),'descend');
    M_hat = Gamma(:,ind(1:R));
%     s2 = sqrt(s2(1:R));
    % F
    [Gamma,S] = eig(Y_3*Y_3');
    [s3,ind] = sort(diag(S),'descend');
    F_hat = Gamma(:,ind(1:R));
%     s3 = sqrt(s3(1:R));
    
    % select correct sign
    for r=1:R
        L_hat(:,r) = L_hat(:,r)*sign(L_hat(:,r)'*L(:,r));
        M_hat(:,r) = M_hat(:,r)*sign(M_hat(:,r)'*M(:,r));
        F_hat(:,r) = F_hat(:,r)*sign(F_hat(:,r)'*F(:,r));
    end

    % estimate error variance
%     Y_hat = double(tensor(ktensor(sqrt(s3(1:R)),{L_hat(:,1:R),M_hat(:,1:R),F_hat(:,1:R)})));
%     Y_e = Y - Y_hat;
%     
%     sig_u_hat = std(Y_e,1,"all");

    % statistic
    Sk_i = NaN(K0-R,1);
    for r=1:K0-R
        Sk_i(r) = (s1(R+r)-s1(R+r+1))/(s1(R+r+1)-s1(R+r+2));
    end
    Sk1(k) = max(Sk_i);

    Sk_i = NaN(K0-R,1);
    for r=1:K0-R
        Sk_i(r) = (s2(R+r)-s2(R+r+1))/(s2(R+r+1)-s2(R+r+2));
    end
    Sk2(k) = max(Sk_i);

    Sk_i = NaN(K0-R,1);
    for r=1:K0-R
        Sk_i(r) = (s3(R+r)-s3(R+r+1))/(s3(R+r+1)-s3(R+r+2));
    end
    Sk3(k) = max(Sk_i);

    % p-values
    p1(k,i) = sum(asydist>Sk1(k))/K;
    p2(k,i) = sum(asydist>Sk2(k))/K;
    p3(k,i) = sum(asydist>Sk3(k))/K;
end

% power(i,1) = sum(Sk1>c)/K;
% power(i,2) = sum(Sk2>c)/K;
% power(i,3) = sum(Sk3>c)/K;

end

power = NaN(length(d2),3);
for i=1:length(d2)
    power(i,1) = sum(p1(:,i)<0.05)/K;
    power(i,2) = sum(p2(:,i)<0.05)/K;
    power(i,3) = sum(p3(:,i)<0.05)/K;
end

% p-value combination
p = NaN(K,length(d2));
power_min = NaN(length(d2),1);power_max = NaN(length(d2),1);
power_mean = NaN(length(d2),1);power_med = NaN(length(d2),1);
for i=1:length(d2)
    
    p(:,i) = min(2*mean([p1(:,i),p2(:,i),p3(:,i)],2),1);
    power_mean(i) = sum(p(:,i)<0.05)/K;
%     p(:,i) = min(exp(1)*geomean([p1(:,i),p2(:,i),p3(:,i)],2),1);
    p(:,i) = max([p1(:,i),p2(:,i),p3(:,i)],[],2);
    power_max(i) = sum(p(:,i)<0.05)/K;
    p(:,i) = min(2*median([p1(:,i),p2(:,i),p3(:,i)],2),1);
    power_med(i) = sum(p(:,i)<0.05)/K;
    p(:,i) = min(3*min([p1(:,i),p2(:,i),p3(:,i)],[],2),1);
    power_min(i) = sum(p(:,i)<0.05)/K;
end

% save('power_3dim.mat','power')

%%
figure;
plot(d2,power(:,2),'-s','LineWidth',2);
hold on
plot(d2,power(:,1),'-s','LineWidth',2);
plot(d2,power(:,3),'-s','LineWidth',2);
plot(d2,power_min,'-s','LineWidth',2,'Color',[0.4940 0.1840 0.5560]);
grid on
set(gca, 'box', 'off')
ylim([0 1.02]);
xL = [-0.01 0.16];
xlim(xL);
line(xL, [0.05,0.05], 'LineWidth', 1, 'Color', [0.4660 0.6740 0.1880],'LineStyle','--');
legend('mat1','mat2','mat3','min','Nominal Level: 5%','Location','northwest');
xlabel('d_2')
ylabel('Empirical rejection probability')
% exportgraphics(gcf,'test_3dim_1.pdf','BackgroundColor','none','ContentType','vector')

%%
figure;
plot(d2,power_max,'-s','LineWidth',2);
hold on
plot(d2,power_mean,'-s','LineWidth',2);
plot(d2,power_med,'-s','LineWidth',2);
plot(d2,power_min,'-s','LineWidth',2,'Color',[0.4940 0.1840 0.5560]);
grid on
set(gca, 'box', 'off')
ylim([0 1.02]);
xL = [-0.01 0.16];
xlim(xL);
line(xL, [0.05,0.05], 'LineWidth', 1, 'Color', [0.4660 0.6740 0.1880],'LineStyle','--');
legend('max','mean','med','min','Nominal Level: 5%','Location','northwest');
xlabel('d_2')
ylabel('Empirical rejection probability')
% exportgraphics(gcf,'test_3dim_2.pdf','BackgroundColor','none','ContentType','vector')


%% different values of K0
power3 = importdata('power_3dim_K3.mat');
power5 = importdata('power_3dim_K5.mat');
power7 = importdata('power_3dim_K7.mat');
%% mat1
figure;
plot(d2,power3(:,2),'-s','LineWidth',2);
hold on
plot(d2,power5(:,2),'-s','LineWidth',2);
plot(d2,power7(:,2),'-s','LineWidth',2);
grid on
set(gca, 'box', 'off')
ylim([0 1.02]);
xL = [-0.01 0.13];
xlim(xL);
line(xL, [0.05,0.05], 'LineWidth', 1, 'Color', [0.4660 0.6740 0.1880],'LineStyle','--');
legend('K=3','K=5','K=7','Nominal Level: 5%','Location','northwest');
xlabel('d_2')
ylabel('Empirical rejection probability')
% exportgraphics(gcf,'test_3dim_mat1.pdf','BackgroundColor','none','ContentType','vector')
%% mat2
figure;
plot(d2,power3(:,1),'-s','LineWidth',2);
hold on
plot(d2,power5(:,1),'-s','LineWidth',2);
plot(d2,power7(:,1),'-s','LineWidth',2);
grid on
set(gca, 'box', 'off')
ylim([0 1.02]);
xL = [-0.01 0.13];
xlim(xL);
line(xL, [0.05,0.05], 'LineWidth', 1, 'Color', [0.4660 0.6740 0.1880],'LineStyle','--');
legend('K=3','K=5','K=7','Nominal Level: 5%','Location','northwest');
xlabel('d_2')
ylabel('Empirical rejection probability')
% exportgraphics(gcf,'test_3dim_mat2.pdf','BackgroundColor','none','ContentType','vector')
%% mat3
figure;
plot(d2,power3(:,3),'-s','LineWidth',2);
hold on
plot(d2,power5(:,3),'-s','LineWidth',2);
plot(d2,power7(:,3),'-s','LineWidth',2);
grid on
set(gca, 'box', 'off')
ylim([0 1.02]);
xL = [-0.01 0.13];
xlim(xL);
line(xL, [0.05,0.05], 'LineWidth', 1, 'Color', [0.4660 0.6740 0.1880],'LineStyle','--');
legend('K=3','K=5','K=7','Nominal Level: 5%','Location','northwest');
xlabel('d_2')
ylabel('Empirical rejection probability')
% exportgraphics(gcf,'test_3dim_mat3.pdf','BackgroundColor','none','ContentType','vector')
