% tensor decomposition via PCA, simulations
clear;clc;
rng(23,'twister');
R = 2;
T = 100;
N = 30;
J = 20;
K = 5000; %# of repetitions
quickstart = 0; %1: PCA starting value for ALS
% generate Lambda
% L = NaN(N,R);
% L(:,1) = 10*(1:N)/N;
% if R>1
%     for r=2:R
%         L(:,r) = N*ones(N,1) - N*ones(N,1)'*L(:,r-1)*L(:,r-1)/(L(:,r-1)'*L(:,r-1));
%     end
% end
% L = L/(sqrtm(L'*L));
L = get_orthonormal(N,R).*sqrt(N);

% generate Mu
% M = NaN(J,R);
% M(:,1) = 10*(1:J)/J;
% if R>1
%     for r=2:R
%         M(:,r) = J*ones(J,1) - J*ones(J,1)'*M(:,r-1)*M(:,r-1)/(M(:,r-1)'*M(:,r-1));
%     end
% end
% M = M/(sqrtm(M'*M));
M = get_orthonormal(J,R).*sqrt(J);

l_1 = NaN(K,R);l_2 = NaN(K,R);l_3 = NaN(K,R);
% l_1_als = NaN(K,R);l_2_als = NaN(K,R);l_3_als = NaN(K,R);
for k = 1:K
    if mod(k, 100) == 0
        disp(k);
    end
% generate F
rho = 0.5;  % this term does not matter?
sig_e = 0.1;
F = NaN(T+100,R);
e = randn(T+100,R);
F(1,:) = sig_e.*e(1,:);
for t=2:(T+100)
    F(t,:) = F(t-1,:).*rho + sig_e .* e(t,:);
end
F = F(101:T+100,:);
F = F/(sqrtm(F'*F/T));

% generate tensor Y   https://www.tensortoolbox.org/ktensor_doc.html#7
sig_u = 0.1;
s = [2;1];
Y = ktensor(s,{L,M,F});
Y = tensor(Y);
U = sig_u*randn([N,J,T]);
Y = double(Y) + U; Y = tensor(Y);
Y_1 = double(reshape(Y,[N,J*T]));
Y_2 = double(reshape(permute(Y,[2,1,3]),[J,N*T]));
Y_3 = double(reshape(permute(Y,[3,1,2]),[T,N*J]));

% tensor decomposition via PCA 
% Lambda
[Gamma_1,S_1] = eig(Y_1*Y_1'/(J*T));
Gamma_1 = Gamma_1*sqrt(N);
L_hat = NaN(N,R);
for r=1:R
    L_hat(:,r) = Gamma_1(:,N-r+1);
end
s_1 = NaN(R,1);
for r=1:R
    s_1(r) = sqrt(S_1(N-r+1,N-r+1));
end
% Mu
[Gamma_2,S_2] = eig(Y_2*Y_2'/(N*T));
Gamma_2 = Gamma_2*sqrt(J);
M_hat = NaN(J,R);
for r=1:R
    M_hat(:,r) = Gamma_2(:,J-r+1);
end
s_2 = NaN(R,1);
for r=1:R
    s_2(r) = sqrt(S_2(J-r+1,J-r+1));
end
% F
[Gamma_3,S_3] = eig(Y_3*Y_3'/(N*J));
Gamma_3 = Gamma_3*sqrt(T);
F_hat = NaN(T,R);
for r=1:R
    F_hat(:,r) = Gamma_3(:,T-r+1);
end
s_3 = NaN(R,1);
for r=1:R
    s_3(r) = sqrt(S_3(T-r+1,T-r+1));
end
% calculate estimation error
for r=1:R
    l_1(k,r) = min([norm(L(:,r)-L_hat(:,r)),norm(L(:,r)-L_hat(:,r).*(-1))]);
    l_2(k,r) = min([norm(M(:,r)-M_hat(:,r)),norm(M(:,r)-M_hat(:,r).*(-1))]);
    l_3(k,r) = min([norm(F(:,r)-F_hat(:,r)),norm(F(:,r)-F_hat(:,r).*(-1))]);
end

% ALS
% if quickstart == 1
%     init = {L_hat,M_hat,F_hat};
%     CPALS = cp_als(Y,R,'printitn',0,'init',init);
% elseif quickstart == 0
%     CPALS = cp_als(Y,R,'printitn',0);
% end
% L_als = CPALS.U{1}*sqrt(N);
% M_als = CPALS.U{2}*sqrt(J);
% F_als = CPALS.U{3}*sqrt(T);
% 
% for r=1:R
%     l_1_als(k,r) = min([norm(L(:,r)-L_als(:,r)),norm(L(:,r)-L_als(:,r).*(-1))]);
%     l_2_als(k,r) = min([norm(M(:,r)-M_als(:,r)),norm(M(:,r)-M_als(:,r).*(-1))]);
%     l_3_als(k,r) = min([norm(F(:,r)-F_als(:,r)),norm(F(:,r)-F_als(:,r).*(-1))]);
% end

end
%% second plot
% rng(23,'twister');
T = 100;
N = 30;
J = 20;
K = 5000; %# of repetitions
quickstart = 0; %1: PCA starting value for ALS
% generate Lambda
% L = NaN(N,R);
% L(:,1) = 10*(1:N)/N;
% if R>1
%     for r=2:R
%         L(:,r) = N*ones(N,1) - N*ones(N,1)'*L(:,r-1)*L(:,r-1)/(L(:,r-1)'*L(:,r-1));
%     end
% end
% L = L/(sqrtm(L'*L/N));
L = get_orthonormal(N,R).*sqrt(N);

% generate Mu
% M = NaN(J,R);
% M(:,1) = 10*(1:J)/J;
% if R>1
%     for r=2:R
%         M(:,r) = J*ones(J,1) - J*ones(J,1)'*M(:,r-1)*M(:,r-1)/(M(:,r-1)'*M(:,r-1));
%     end
% end
% M = M/(sqrtm(M'*M/J));
M = get_orthonormal(J,R).*sqrt(J);

l_11 = NaN(K,R);l_22 = NaN(K,R);l_33 = NaN(K,R);
for k = 1:K
    if mod(k, 100) == 0
        disp(k);
    end
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
F = F/(sqrtm(F'*F/T));

% generate tensor Y   https://www.tensortoolbox.org/ktensor_doc.html#7
sig_u = 0.1;
s = [4;2];
Y = ktensor(s,{L,M,F});
Y = tensor(Y);
U = sig_u*randn([N,J,T]);
Y = double(Y) + U; Y = tensor(Y);
Y_1 = double(reshape(Y,[N,J*T]));
Y_2 = double(reshape(permute(Y,[2,1,3]),[J,N*T]));
Y_3 = double(reshape(permute(Y,[3,1,2]),[T,N*J]));

% tensor decomposition via PCA
% Lambda
[Gamma_1,S_1] = eig(Y_1*Y_1'/(J*T));
Gamma_1 = Gamma_1*sqrt(N);
L_hat = NaN(N,R);
for r=1:R
    L_hat(:,r) = Gamma_1(:,N-r+1);
end
s_1 = NaN(R,1);
for r=1:R
    s_1(r) = sqrt(S_1(N-r+1,N-r+1));
end
% Mu
[Gamma_2,S_2] = eig(Y_2*Y_2'/(N*T));
Gamma_2 = Gamma_2*sqrt(J);
M_hat = NaN(J,R);
for r=1:R
    M_hat(:,r) = Gamma_2(:,J-r+1);
end
s_2 = NaN(R,1);
for r=1:R
    s_2(r) = sqrt(S_2(J-r+1,J-r+1));
end
% F
[Gamma_3,S_3] = eig(Y_3*Y_3'/(N*J));
Gamma_3 = Gamma_3*sqrt(T);
F_hat = NaN(T,R);
for r=1:R
    F_hat(:,r) = Gamma_3(:,T-r+1);
end
s_3 = NaN(R,1);
for r=1:R
    s_3(r) = sqrt(S_3(T-r+1,T-r+1));
end
% calculate estimation error
for r=1:R
    l_11(k,r) = min([norm(L(:,r)-L_hat(:,r)),norm(L(:,r)-L_hat(:,r).*(-1))]);
    l_22(k,r) = min([norm(M(:,r)-M_hat(:,r)),norm(M(:,r)-M_hat(:,r).*(-1))]);
    l_33(k,r) = min([norm(F(:,r)-F_hat(:,r)),norm(F(:,r)-F_hat(:,r).*(-1))]);
end

end


%%
figure
for r=1:R
    ax1(r)=subplot(3,R,r);
    histogram(l_1(:,r));
    hold on
    histogram(l_11(:,r));
    yL=get(gca,'YLim');
    line([mean(l_1(:,r)), mean(l_1(:,r))], yL, 'LineWidth', 1, 'Color', 'r');
    text(mean(l_1(:,r)),yL(2),num2str(round(mean(l_1(:,r)),2,'significant')),'VerticalAlignment','top'); 
    line([mean(l_11(:,r)), mean(l_11(:,r))], yL, 'LineWidth', 1, 'Color', 'r');
    text(mean(l_11(:,r)),yL(2),num2str(round(mean(l_11(:,r)),2,'significant')),'VerticalAlignment','top','HorizontalAlignment','right');
    title(['\lambda_',num2str(r)]);
    ylim(yL);
    ax2(r)=subplot(3,R,r+R);
    histogram(l_2(:,r));
    hold on
    histogram(l_22(:,r));
    yL=get(gca,'YLim');
    line([mean(l_2(:,r)), mean(l_2(:,r))], yL, 'LineWidth', 1, 'Color', 'r');
    text(mean(l_2(:,r)),yL(2),num2str(round(mean(l_2(:,r)),2,'significant')),'VerticalAlignment','top');
    line([mean(l_22(:,r)), mean(l_22(:,r))], yL, 'LineWidth', 1, 'Color', 'r');
    text(mean(l_22(:,r)),yL(2),num2str(round(mean(l_22(:,r)),2,'significant')),'VerticalAlignment','top','HorizontalAlignment','right');
    title(['\mu_',num2str(r)]);
    ylim(yL);
    ax3(r)=subplot(3,R,r+2*R);
    histogram(l_3(:,r));
    hold on
    histogram(l_33(:,r));
    yL=get(gca,'YLim');
    line([mean(l_3(:,r)), mean(l_3(:,r))], yL, 'LineWidth', 1, 'Color', 'r');
    text(mean(l_3(:,r)),yL(2),num2str(round(mean(l_3(:,r)),2,'significant')),'VerticalAlignment','top');
    line([mean(l_33(:,r)), mean(l_33(:,r))], yL, 'LineWidth', 1, 'Color', 'r');
    text(mean(l_33(:,r)),yL(2),num2str(round(mean(l_33(:,r)),2,'significant')),'VerticalAlignment','top','HorizontalAlignment','right');
    title(['f_',num2str(r)]);
    ylim(yL);
end
set(gcf, 'Position',  [500, 500, 600, 500])

