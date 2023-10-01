% TW distributed test

clear;clc;
rng(23,'twister');
R0 = 2; % true rank
R = 1; %rank estimated
T = 50;
N1 = 10;
N2 = 20;
N3 = 30;
N4 = 40;
K = 5000; %# of repetitions

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
% save('dist_K5.mat','asydist')

% generate loadings
V1 = get_orthonormal(N1,R0);

V2 = get_orthonormal(N2,R0);

V3 = get_orthonormal(N3,R0);

V4 = get_orthonormal(N4,R0);


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

d2=[0:0.01:0.15]; %0.4 for normal, 0.5 for t
% d2 = 0;
% power1 = NaN(length(d2),1);power2 = NaN(length(d2),1);power3 = NaN(length(d2),1);
p1 = NaN(K,length(d2));p2 = NaN(K,length(d2));p3 = NaN(K,length(d2));p4 = NaN(K,length(d2));p5 = NaN(K,length(d2));
for i=1:length(d2)
    disp(d2(i));

% signal strength
s = sqrt(N1*N2*N3*N4*T)*([2,d2(i)])';D=diag(s);

Sk1 = NaN(K,1);Sk2 = NaN(K,1);Sk3 = NaN(K,1);Sk4 = NaN(K,1);Sk5 = NaN(K,1);
for k = 1:K

    if mod(k, 100) == 0
        disp(k);
    end
    
    % generate tensor Y   https://www.tensortoolbox.org/ktensor_doc.html#7
    sig_u = 1;
    % s = [2*sqrt((N*J*T)^a1);*sqrt((N*J*T)^a2)];
    Y = ktensor(s,{V1,V2,V3,V4,F});
    Y = tensor(Y);
    U = sig_u*randn([N1,N2,N3,N4,T]);
%     U = sig_u*sqrt(3/5)*trnd(5,N,J,T); % t-dist errors
    Y = double(Y) + U;
    Y_1 = reshape(Y,[N1,N2*N3*N4*T]);
    Y_2 = reshape(permute(Y,[2,1,3,4,5]),[N2,N1*N3*N4*T]);
    Y_3 = reshape(permute(Y,[3,1,2,4,5]),[N3,N1*N2*N4*T]);
    Y_4 = reshape(permute(Y,[4,1,2,3,5]),[N4,N1*N2*N3*T]);
    Y_5 = reshape(permute(Y,[5,1,2,3,4]),[T,N1*N2*N3*N4]);

    % tensor decomposition via PCA
    % Lambda
    [Gamma,S] = eig(Y_1*Y_1');
    [s1,ind] = sort(diag(S),'descend');
    V1_hat = Gamma(:,ind(1:R));
    
    [Gamma,S] = eig(Y_2*Y_2');
    [s2,ind] = sort(diag(S),'descend');
    V2_hat = Gamma(:,ind(1:R));
    
    [Gamma,S] = eig(Y_3*Y_3');
    [s3,ind] = sort(diag(S),'descend');
    V3_hat = Gamma(:,ind(1:R));

    [Gamma,S] = eig(Y_4*Y_4');
    [s4,ind] = sort(diag(S),'descend');
    V4_hat = Gamma(:,ind(1:R));

    [Gamma,S] = eig(Y_5*Y_5');
    [s5,ind] = sort(diag(S),'descend');
    F_hat = Gamma(:,ind(1:R));
    
    % select correct sign
    for r=1:R
        V1_hat(:,r) = V1_hat(:,r)*sign(V1_hat(:,r)'*V1(:,r));
        V2_hat(:,r) = V2_hat(:,r)*sign(V2_hat(:,r)'*V2(:,r));
        V3_hat(:,r) = V3_hat(:,r)*sign(V3_hat(:,r)'*V3(:,r));
        V4_hat(:,r) = V4_hat(:,r)*sign(V4_hat(:,r)'*V4(:,r));
        F_hat(:,r) = F_hat(:,r)*sign(F_hat(:,r)'*F(:,r));
    end

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

    Sk_i = NaN(K0-R,1);
    for r=1:K0-R
        Sk_i(r) = (s4(R+r)-s4(R+r+1))/(s4(R+r+1)-s4(R+r+2));
    end
    Sk4(k) = max(Sk_i);

    Sk_i = NaN(K0-R,1);
    for r=1:K0-R
        Sk_i(r) = (s5(R+r)-s5(R+r+1))/(s5(R+r+1)-s5(R+r+2));
    end
    Sk5(k) = max(Sk_i);

    % p-values
    p1(k,i) = sum(asydist>Sk1(k))/K;
    p2(k,i) = sum(asydist>Sk2(k))/K;
    p3(k,i) = sum(asydist>Sk3(k))/K;
    p4(k,i) = sum(asydist>Sk4(k))/K;
    p5(k,i) = sum(asydist>Sk5(k))/K;

end

% power1(i) = sum(d1)/K;
% power2(i) = sum(d2)/K;
% power3(i) = sum(d3)/K;

end

power = NaN(length(d2),5);
for i=1:length(d2)
    power(i,1) = sum(p1(:,i)<0.05)/K;
    power(i,2) = sum(p2(:,i)<0.05)/K;
    power(i,3) = sum(p3(:,i)<0.05)/K;
    power(i,4) = sum(p4(:,i)<0.05)/K;
    power(i,5) = sum(p5(:,i)<0.05)/K;
end

% p-value combination
p = NaN(K,length(d2));
power_min = NaN(length(d2),1);power_max = NaN(length(d2),1);
power_mean = NaN(length(d2),1);power_med = NaN(length(d2),1);
for i=1:length(d2)
    
    p(:,i) = min(2*mean([p1(:,i),p2(:,i),p3(:,i),p4(:,i),p5(:,i)],2),1);
    power_mean(i) = sum(p(:,i)<0.05)/K;
%     p(:,i) = min(exp(1)*geomean([p1(:,i),p2(:,i),p3(:,i),p4(:,i),p5(:,i)],2),1);
    p(:,i) = max([p1(:,i),p2(:,i),p3(:,i),p4(:,i),p5(:,i)],[],2);
    power_max(i) = sum(p(:,i)<0.05)/K;
    p(:,i) = min(2*median([p1(:,i),p2(:,i),p3(:,i),p4(:,i),p5(:,i)],2),1);
    power_med(i) = sum(p(:,i)<0.05)/K;
    p(:,i) = min(5*min([p1(:,i),p2(:,i),p3(:,i),p4(:,i),p5(:,i)],[],2),1);
    power_min(i) = sum(p(:,i)<0.05)/K;
end

% save('pvals_5dim.mat','p1','p2','p3','p4','p5')

%% 
figure;
plot(d2(1:10),power(1:10,1),'-s','LineWidth',2);
hold on
plot(d2(1:10),power(1:10,2),'-s','LineWidth',2);
plot(d2(1:10),power(1:10,3),'-s','LineWidth',2);
plot(d2(1:10),power(1:10,4),'-s','LineWidth',2);
plot(d2(1:10),power(1:10,5),'-s','LineWidth',2);
plot(d2(1:10),power_min(1:10),'-s','LineWidth',2,'Color',[0.3010 0.7450 0.9330]);
grid on
set(gca, 'box', 'off')
ylim([0 1.02]);
xL = [-0.01 0.1];
xlim(xL);
line(xL, [0.05,0.05], 'LineWidth', 1, 'Color', [0.4660 0.6740 0.1880],'LineStyle','--');
legend('mat1','mat2','mat3','mat4','mat5','min','Nominal Level: 5%','Location','northwest');
xlabel('d_2')
ylabel('Empirical rejection probability')
% exportgraphics(gcf,'test_5dim_1.pdf','BackgroundColor','none','ContentType','vector')


%%
figure;
plot(d2(1:10),power_max(1:10),'-s','LineWidth',2);
hold on
plot(d2(1:10),power_mean(1:10),'-s','LineWidth',2);
plot(d2(1:10),power_med(1:10),'-s','LineWidth',2);
plot(d2(1:10),power_min(1:10),'-s','LineWidth',2,'Color',[0.3010 0.7450 0.9330]);
grid on
set(gca, 'box', 'off')
ylim([0 1.02]);
xL = [-0.01 0.1];
xlim(xL);
line(xL, [0.05,0.05], 'LineWidth', 1, 'Color', [0.4660 0.6740 0.1880],'LineStyle','--');
legend('max','mean','med','min','Nominal Level: 5%','Location','northwest');
xlabel('d_2')
ylabel('Empirical rejection probability')
% exportgraphics(gcf,'test_5dim_2.pdf','BackgroundColor','none','ContentType','vector')
