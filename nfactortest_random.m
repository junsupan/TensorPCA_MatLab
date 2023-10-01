%% random sample split

clear;clc;
rng(23,'twister');
R0 = 2; % true rank
T = 50;
N = 40;
J = 30;
K = 5000; %# of repetitions


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

d2=[0:0.05:1]; %0.4 for normal, 0.5 for t
% d2 = 0;
% power1 = NaN(length(d2),1);power2 = NaN(length(d2),1);power3 = NaN(length(d2),1);
p1 = NaN(K,length(d2));p2 = NaN(K,length(d2));p3 = NaN(K,length(d2));
for i=1:length(d2)
    disp(d2(i));

% signal strength
s = sqrt(N*J*T)*([2,d2(i)])';D=diag(s);

R = 1; %rank estimated



% d1 = NaN(K,1);d2 = NaN(K,1);d3 = NaN(K,1);
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

    % random sample splitting
    ind1 = datasample(1:N*J,N*J/2,'Replace',false);
    ind1 = sort(ind1);
    ind2 = setdiff(1:N*J,ind1);

    Y_31 = Y_3(:,ind1);
    [~,S] = eig(Y_31*Y_31');
    [s31,~] = sort(diag(S),'descend');
    
    Y_32 = Y_3(:,ind2);
    [~,S] = eig(Y_32*Y_32');
    [s32,~] = sort(diag(S),'descend');

    % estimate error variance
    Y_hat = double(tensor(ktensor(sqrt(s3(1:R)),{L_hat(:,1:R),M_hat(:,1:R),F_hat(:,1:R)})));
    Y_e3 = Y - Y_hat;
    Y_e3 = reshape(permute(Y_e3,[3,1,2]),[T,N*J]);
    Y_e3 = Y_e3(:,ind2);
%     Y_e3 = Y_e3(:,ind1);
    
    sig_u_hat = std(Y_e3,1,"all");
    
    mean_u = mean(Y_e3,"all");

    gamma_4 = mean((Y_e3 - mean_u).^4,"all");
    
    var_Sk = 2*(gamma_4 - sig_u_hat^4);

    % statistic
    Sk3(k) = sum(s31(R+1:end))/sqrt(N*J*(T-R)/2)-sig_u_hat^2*sqrt(N*J*(T-R)/2);
%     Sk(k) = sum(s31(R+1:end))/sqrt(N*J*(T-R)/2)-sum(s32(R+1:end))/sqrt(N*J*(T-R)/2);
    
%     c = 2*sig_u_hat^2*norminv(0.975); % critical value
    c = sqrt(var_Sk) * norminv(0.975);
    p3(k,i) = 2*(1-normcdf(abs(Sk3(k)/sqrt(var_Sk))));

%     if Sk3(k)>c || Sk3(k)<(-c)
%         d3(k) = 1;
%     else
%         d3(k) = 0;
%     end

    % random sample splitting
    ind1 = datasample(1:J*T,J*T/2,'Replace',false);
    ind1 = sort(ind1);
    ind2 = setdiff(1:J*T,ind1);

    Y_11 = Y_1(:,ind1);
    [~,S] = eig(Y_11*Y_11');
    [s11,~] = sort(diag(S),'descend');
    
    Y_12 = Y_1(:,ind2);
    [~,S] = eig(Y_12*Y_12');
    [s12,~] = sort(diag(S),'descend');

    % estimate error variance
    Y_hat = double(tensor(ktensor(sqrt(s1(1:R)),{L_hat(:,1:R),M_hat(:,1:R),F_hat(:,1:R)})));
    Y_e1 = Y - Y_hat;
    Y_e1 = reshape(Y_e1,[N,J*T]);
    Y_e1 = Y_e1(:,ind2);
%     Y_e1 = Y_e1(:,ind1);
    
    sig_u_hat = std(Y_e1,1,"all");
    
    mean_u = mean(Y_e1,"all");

    gamma_4 = mean((Y_e1 - mean_u).^4,"all");
    
    var_Sk = 2*(gamma_4 - sig_u_hat^4);

    % statistic
    Sk1(k) = sum(s11(R+1:end))/sqrt(T*J*(N-R)/2)-sig_u_hat^2*sqrt(T*J*(N-R)/2);
%     Sk1(k) = sum(s11(R+1:end))/sqrt(T*J*(N-R)/2)-sum(s12(R+1:end))/sqrt(T*J*(N-R)/2);
    
%     c = 2*sig_u_hat^2*norminv(0.975); % critical value
    c = sqrt(var_Sk) * norminv(0.975);
    p1(k,i) = 2*(1-normcdf(abs(Sk1(k)/sqrt(var_Sk))));

%     if Sk1(k)>c || Sk1(k)<(-c)
%         d1(k) = 1;
%     else
%         d1(k) = 0;
%     end

    % random sample splitting
    ind1 = datasample(1:N*T,N*T/2,'Replace',false);
    ind1 = sort(ind1);
    ind2 = setdiff(1:N*T,ind1);

    Y_21 = Y_2(:,ind1);
    [~,S] = eig(Y_21*Y_21');
    [s21,~] = sort(diag(S),'descend');
    
    Y_22 = Y_2(:,ind2);
    [~,S] = eig(Y_22*Y_22');
    [s22,~] = sort(diag(S),'descend');

    % estimate error variance
    Y_hat = double(tensor(ktensor(sqrt(s2(1:R)),{L_hat(:,1:R),M_hat(:,1:R),F_hat(:,1:R)})));
    Y_e2 = Y - Y_hat;
    Y_e2 = reshape(permute(Y_e2,[2,1,3]),[J,N*T]);
    Y_e2 = Y_e2(:,ind2);
%     Y_e2 = Y_e2(:,ind1);
    
    sig_u_hat = std(Y_e2,1,"all");
    
    mean_u = mean(Y_e2,"all");

    gamma_4 = mean((Y_e2 - mean_u).^4,"all");
    
    var_Sk = 2*(gamma_4 - sig_u_hat^4);

    % statistic
    Sk2(k) = sum(s21(R+1:end))/sqrt(N*T*(J-R)/2)-sig_u_hat^2*sqrt(N*T*(J-R)/2);
%     Sk2(k) = sum(s21(R+1:end))/sqrt(N*T*(J-R)/2)-sum(s22(R+1:end))/sqrt(N*T*(J-R)/2);
    
%     c = 2*sig_u_hat^2*norminv(0.975); % critical value
    c = sqrt(var_Sk) * norminv(0.975);
    p2(k,i) = 2*(1-normcdf(abs(Sk2(k)/sqrt(var_Sk))));

%     if Sk2(k)>c || Sk2(k)<(-c)
%         d2(k) = 1;
%     else
%         d2(k) = 0;
%     end

end

% power1(i) = sum(d1)/K;
% power2(i) = sum(d2)/K;
% power3(i) = sum(d3)/K;

end

% save('power3.mat','power')

p = NaN(K,length(d2),3);
p(:,:,1) = p1;p(:,:,2) = p2;p(:,:,3) = p3;
p_med = 2*median(p,3);
power_med = sum(p_med<0.05,1)./K;
p_max = max(p,[],3);
power_max = sum(p_max<0.05,1)./K;
p_min = min(p,[],3);
power_min = sum(p_min<0.05,1)./K;
p_mean = mean(p,3);
power_mean = sum(p_mean<0.05,1)./K;

%%
figure;
plot(d2,power_med,'-s','LineWidth',2);
hold on
plot(d2,power_mean,'-s','LineWidth',2);
plot(d2,power_min,'-s','LineWidth',2);
plot(d2,power_max,'-s','LineWidth',2);
grid on
set(gca, 'box', 'off')
ylim([0 1.02]);
xL = [-0.02 1.02];
xlim(xL);
line(xL, [0.05,0.05], 'LineWidth', 1, 'Color', [0.4660 0.6740 0.1880],'LineStyle','--');
legend('median','mean','min','max','Location','northwest');
xlabel('d_2')
ylabel('Empirical rejection probability')
% exportgraphics(gcf,'testquantiles.pdf','BackgroundColor','none','ContentType','vector')