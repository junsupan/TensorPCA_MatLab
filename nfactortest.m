% test for the number of factors
clear;clc;
rng(23,'twister');
R = 1;
T = 100;
N = 60;
J = 40;
K = 2000; %# of repetitions

% generate Lambda
L = get_orthonormal(N,R);
% L_norm = L/(sqrtm(L'*L));

% generate Mu
M = get_orthonormal(J,R);
% M_norm = M/(sqrtm(M'*M));

% generate F
rho = 0.5;
sig_e = 0.1;
F = NaN(T+100,R);
e = randn(T+100,R);
F(1,:) = sig_e.*e(1,:);
for t=2:(T+100)
    F(t,:) = F(t-1,:).*rho + sig_e .* e(t,:);
end
F = F(101:T+100,:);
F = F/(sqrtm(F'*F));
% F_norm = F/(sqrtm(F'*F));

% signal strength
s = sqrt(N*J*T)*(R:-1:1)';D=diag(s);

R = 1; %rank estimated
d = NaN(K,1);
Sk = NaN(K,1);
test = NaN(K,1);
for k = 1:K
    if mod(k, 100) == 0
        disp(k);
    end
    % generate tensor Y   https://www.tensortoolbox.org/ktensor_doc.html#7
    sig_u = 1;
    % s = [2*sqrt((N*J*T)^a1);*sqrt((N*J*T)^a2)];
    Y = ktensor(s,{L,M,F});
    Y = tensor(Y);
    U = sig_u*randn([N,J,T]);
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
    Y_hat = double(tensor(ktensor(sqrt(s3(1:R)),{L_hat(:,1:R),M_hat(:,1:R),F_hat(:,1:R)})));
    Y_e = Y - Y_hat;
    sig_u_hat = std(Y_e,1,"all");
    
    % statistic
%     Sk(k) = sum(s3(R+1:end))/sqrt(N*J)-sig_u_hat^2*N*J*(T-R)/sqrt(N*J);
%     Sk(k) = sum(s3(R+1:end))/sqrt(N*J*T)-sig_u^2*N*J*(T-R)/sqrt(N*J*T);
    Sk(k) = sig_u_hat^2*sqrt(N*J*(T-R));
    test(k) = sqrt(N*J*(T-R))*(sig_u_hat^2-sig_u^2);

%     c = sqrt(2)*sig_u^2*1.96; % reject region
% 
%     if Sk(k)>c || Sk(k)<(-c)
%         d(k) = 1;
%     else
%         d(k) = 0;
%     end
end

histogram(Sk)

std(Sk)


%% sample split

clear;clc;
rng(23,'twister');
R0 = 2; % true rank
T = 100;
N = 80;
J = 60;
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

% d2=[0:0.05:0.5]; %0.4 for normal, 0.5 for t
d2 = 1;
power = NaN(length(d2),1);
for i=1:length(d2)
    disp(d2(i));

% signal strength
s = sqrt(N*J*T)*([2,d2(i)])';D=diag(s);

R = 1; %rank estimated
d = NaN(K,1);
Sk = NaN(K,1);

for k = 1:K
    if mod(k, 100) == 0
        disp(k);
    end
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

    % sample splitting
    Y_31 = Y_3(:,1:(N*J/2));
    [~,S] = eig(Y_31*Y_31');
    [s31,~] = sort(diag(S),'descend');
    
    Y_32 = Y_3(:,(N*J/2)+1:end);
    [~,S] = eig(Y_32*Y_32');
    [s32,~] = sort(diag(S),'descend');

    % estimate error variance
    Y_hat = double(tensor(ktensor(sqrt(s3(1:R)),{L_hat(:,1:R),M_hat(:,1:R),F_hat(:,1:R)})));
    Y_e = Y - Y_hat;
    Y_e = reshape(permute(Y_e,[3,1,2]),[T,N*J]);
    Y_e = Y_e(:,(N*J/2)+1:end);
%     Y_e = Y_e(:,1:(N*J/2));
    
    sig_u_hat = std(Y_e,1,"all");
    
    mean_u = mean(Y_e,"all");

    gamma_4 = mean((Y_e - mean_u).^4,"all");
    
    var_Sk = 2*(gamma_4 - sig_u_hat^4);

    % statistic
%     Sk(k) = sum(s31(R+1:end))/sqrt(T*N*J/2)-sig_u_hat^2*N*J*(T-R)/sqrt(T*N*J*2);
    Sk(k) = sum(s31(R+1:end))/sqrt(N*J*(T-R)/2)-sig_u_hat^2*sqrt(N*J*(T-R)/2);
%     Sk(k) = sum(s31(R+1:end))/sqrt(N*J*(T-R)/2)-sum(s32(R+1:end))/sqrt(N*J*(T-R)/2);
    
%     c = 2*sig_u_hat^2*norminv(0.975); % critical value
    c = sqrt(var_Sk) * norminv(0.975);

    if Sk(k)>c || Sk(k)<(-c)
        d(k) = 1;
    else
        d(k) = 0;
    end

end

power(i) = sum(d)/K;

end

% save('power.mat','power')

% var(Y_3(:,1:N*J/2),1,'all')
% var(Y_3(:,N*J/2+1:end),1,'all')

%% random sample split

clear;clc;
rng(23,'twister');
R0 = 2; % true rank
T = 100;
N = 80;
J = 60;
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

% d2=[0:0.05:0.5]; %0.4 for normal, 0.5 for t
d2 = 1;
power = NaN(length(d2),1);
for i=1:length(d2)
    disp(d2(i));

% signal strength
s = sqrt(N*J*T)*([2,d2(i)])';D=diag(s);

R = 1; %rank estimated

d = NaN(K,1);
Sk = NaN(K,1);Sk0 = inf;
p = NaN(K,1);
for k = 1:K

    if mod(k, 100) == 0
        disp(k);
    end
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

    % deterministic
%     ind1 = 1:N*J/2;
%     ind2 = N*J/2+1:N*J;

    Y_31 = Y_3(:,ind1);
    [~,S] = eig(Y_31*Y_31');
    [s31,~] = sort(diag(S),'descend');
    
    Y_32 = Y_3(:,ind2);
    [~,S] = eig(Y_32*Y_32');
    [s32,~] = sort(diag(S),'descend');

    % estimate error variance
    Y_hat = double(tensor(ktensor(sqrt(s3(1:R)),{L_hat(:,1:R),M_hat(:,1:R),F_hat(:,1:R)})));
    Y_e = Y - Y_hat;
    Y_e = reshape(permute(Y_e,[3,1,2]),[T,N*J]);
    Y_e = Y_e(:,ind2);
%     Y_e = Y_e(:,ind1);
    
    sig_u_hat = std(Y_e,1,"all");
    
    mean_u = mean(Y_e,"all");

    gamma_4 = mean((Y_e - mean_u).^4,"all");
    
    var_Sk = 2*(gamma_4 - sig_u_hat^4);

    % statistic
%     Sk(k) = sum(s31(R+1:end))/sqrt(T*N*J/2)-sig_u_hat^2*N*J*(T-R)/sqrt(T*N*J*2);
    Sk(k) = sum(s31(R+1:end))/sqrt(N*J*(T-R)/2)-sig_u_hat^2*sqrt(N*J*(T-R)/2);
%     Sk(k) = sum(s31(R+1:end))/sqrt(N*J*(T-R)/2)-sum(s32(R+1:end))/sqrt(N*J*(T-R)/2);
    
%     c = 2*sig_u_hat^2*norminv(0.975); % critical value
    c = sqrt(var_Sk) * norminv(0.975);
    p(k) = 2*(1-normcdf(abs(Sk(k)/sqrt(var_Sk))));

    if Sk(k)>c || Sk(k)<(-c)
        d(k) = 1;
    else
        d(k) = 0;
    end

    %%%%%%
    if abs(Sk(k)) <= abs(Sk0)
        Sk0 = Sk(k);
        L0 = L(:,2);
        M0 = M(:,2);
        ind10 = ind1;
        ind20 = ind2;
    end


end

power(i) = sum(d)/K;

end

loadings = kron(M0,L0);
norm(loadings(ind10));
norm(loadings(ind20));

% save('power3.mat','power')

%% 
% power=importdata('powert.mat');
% power2=importdata('powert2.mat');

figure;
plot(d2,power,'-s','LineWidth',2, 'Color',[0 0.4470 0.7410]);
hold on
% plot(d2,power2,'-s','LineWidth',2, 'Color',[0.8500 0.3250 0.0980]);
grid on
set(gca, 'box', 'off')
ylim([0 1.02]);
xL = [-0.02 0.52];
xlim(xL);
line(xL, [0.05,0.05], 'LineWidth', 1, 'Color', [0.4660 0.6740 0.1880],'LineStyle','--');
% legend('50x30x20','100x60x40','Nominal Level: 5%','Location','northwest');
xlabel('d_2')
ylabel('Empirical rejection probability')
% exportgraphics(gcf,'testnormal.pdf','BackgroundColor','none','ContentType','vector')
