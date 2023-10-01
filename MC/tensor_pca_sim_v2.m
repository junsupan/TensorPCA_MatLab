% tensor decomposition via PCA, simulations
% finite sample properties
clear;clc;
rng(23,'twister');
alpha1 = 1;
alpha2 = 1;
alpha3 = 1;
R = 1;
T = 100;
N = 30;
J = 20;
K = 5000; %# of repetitions
quickstart = 1; %1: PCA starting value for ALS
% generate Lambda
L = get_orthonormal(N,R);
L_norm = L/(sqrtm(L'*L));

% generate Mu
M = get_orthonormal(J,R);
M_norm = M/(sqrtm(M'*M));

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
F_norm = F/(sqrtm(F'*F));

% signal strength
s = sqrt(N^alpha1*J^alpha2*T^alpha3)*(R:-1:1)';D=diag(s);

e_l1 = NaN(K,R);e_m1 = NaN(K,R);e_f1 = NaN(K,R);
% l_1_als = NaN(K,R);l_2_als = NaN(K,R);l_3_als = NaN(K,R);
for k = 1:K
    if mod(k, 100) == 0
        disp(k);
    end

% generate tensor Y   https://www.tensortoolbox.org/ktensor_doc.html#7
sig_u = 1;
Y = ktensor(s,{L,M,F});
Y = tensor(Y);
U = sig_u*randn([N,J,T]);
Y = double(Y) + U;
Y_1 = reshape(Y,[N,J*T]);
Y_2 = reshape(permute(Y,[2,1,3]),[J,N*T]);
Y_3 = reshape(permute(Y,[3,1,2]),[T,N*J]);

% tensor decomposition via PCA 
% Lambda
[Gamma_1,S_1] = eig(Y_1*Y_1');
% Gamma_1 = Gamma_1*sqrt(N);
[s_1,ind] = sort(diag(S_1),'descend');
L_hat = Gamma_1(:,ind(1:R));
% Mu
[Gamma_2,S_2] = eig(Y_2*Y_2');
% Gamma_2 = Gamma_2*sqrt(J);
[s_2,ind] = sort(diag(S_2),'descend');
M_hat = Gamma_2(:,ind(1:R));
% F
[Gamma_3,S_3] = eig(Y_3*Y_3');
% Gamma_3 = Gamma_3*sqrt(T);
[s_3,ind] = sort(diag(S_3),'descend');
F_hat = Gamma_3(:,ind(1:R));
% calculate estimation error
for r=1:R
    e_l1(k,r) = norm(L_norm(:,r)-L_hat(:,r)*sign(L_hat(:,r)'*L_norm(:,r)));
    e_m1(k,r) = norm(M_norm(:,r)-M_hat(:,r)*sign(M_hat(:,r)'*M_norm(:,r)));
    e_f1(k,r) = norm(F_norm(:,r)-F_hat(:,r)*sign(F_hat(:,r)'*F_norm(:,r)));
end

% for r=1:R
%     l_1(k,r) = min([norm(L_norm(:,r)-L_hat(:,r)),norm(L_norm(:,r)-L_hat(:,r).*(-1))]);
%     l_2(k,r) = min([norm(M_norm(:,r)-M_hat(:,r)),norm(M_norm(:,r)-M_hat(:,r).*(-1))]);
%     l_3(k,r) = min([norm(F_norm(:,r)-F_hat(:,r)),norm(F_norm(:,r)-F_hat(:,r).*(-1))]);
% end

end

%% second plot
% rng(23,'twister');
T = 100;
N = 60;
J = 20;
% generate Lambda
L = get_orthonormal(N,R);
L_norm = L/(sqrtm(L'*L));

% generate Mu
M = get_orthonormal(J,R);
M_norm = M/(sqrtm(M'*M));

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
F_norm = F/(sqrtm(F'*F));

% signal strength
s = sqrt(N^alpha1*J^alpha2*T^alpha3)*(R:-1:1)';D=diag(s);

e_l2 = NaN(K,R);e_m2 = NaN(K,R);e_f2 = NaN(K,R);
for k = 1:K
    if mod(k, 100) == 0
        disp(k);
    end

% generate tensor Y   https://www.tensortoolbox.org/ktensor_doc.html#7
sig_u = 1;
Y = ktensor(s,{L,M,F});
Y = tensor(Y);
U = sig_u*randn([N,J,T]);
Y = double(Y) + U;
Y_1 = reshape(Y,[N,J*T]);
Y_2 = reshape(permute(Y,[2,1,3]),[J,N*T]);
Y_3 = reshape(permute(Y,[3,1,2]),[T,N*J]);

% tensor decomposition via PCA
% Lambda
[Gamma_1,S_1] = eig(Y_1*Y_1');
% Gamma_1 = Gamma_1*sqrt(N);
[s_1,ind] = sort(diag(S_1),'descend');
L_hat = Gamma_1(:,ind(1:R));
% Mu
[Gamma_2,S_2] = eig(Y_2*Y_2');
% Gamma_2 = Gamma_2*sqrt(J);
[s_2,ind] = sort(diag(S_2),'descend');
M_hat = Gamma_2(:,ind(1:R));
% F
[Gamma_3,S_3] = eig(Y_3*Y_3');
% Gamma_3 = Gamma_3*sqrt(T);
[s_3,ind] = sort(diag(S_3),'descend');
F_hat = Gamma_3(:,ind(1:R));
% calculate estimation error
for r=1:R
    e_l2(k,r) = norm(L_norm(:,r)-L_hat(:,r)*sign(L_hat(:,r)'*L_norm(:,r)));
    e_m2(k,r) = norm(M_norm(:,r)-M_hat(:,r)*sign(M_hat(:,r)'*M_norm(:,r)));
    e_f2(k,r) = norm(F_norm(:,r)-F_hat(:,r)*sign(F_hat(:,r)'*F_norm(:,r)));
end
% for r=1:R
%     l_11(k,r) = min([norm(L_norm(:,r)-L_hat(:,r)),norm(L_norm(:,r)-L_hat(:,r).*(-1))]);
%     l_22(k,r) = min([norm(M_norm(:,r)-M_hat(:,r)),norm(M_norm(:,r)-M_hat(:,r).*(-1))]);
%     l_33(k,r) = min([norm(F_norm(:,r)-F_hat(:,r)),norm(F_norm(:,r)-F_hat(:,r).*(-1))]);
% end

end

%%
e_1 = e_f1;e_2 = e_f2;
r=1;
figure
ax = subplot(1,1,1);
histogram(e_1(:,r));
hold on
histogram(e_2(:,r));
xline(mean(e_1(:,r)),':',num2str(round(mean(e_1(:,r)),2,'significant')));
xline(mean(e_2(:,r)),':',num2str(round(mean(e_2(:,r)),2,'significant')));
xlabel('Estimation Error')
ylabel('Count')
legend('baseline','modified')
ax.YGrid = 'on';
ylim([0 550])
% exportgraphics(gcf,'sim_l1.pdf','BackgroundColor','none','ContentType','vector')

%% in slides

e_1 = e_m1;e_2 = e_m2;
r=1;
figure
ax = subplot(1,1,1);
histogram(e_1(:,r));
hold on
histogram(e_2(:,r));
line1=xline(mean(e_1(:,r)),':',num2str(round(mean(e_1(:,r)),2,'significant')),LineWidth=2);
line2=xline(mean(e_2(:,r)),':',num2str(round(mean(e_2(:,r)),2,'significant')),LineWidth=2);
xlab=xlabel('Estimation Error');
% ylabel('Count')
lgd=legend('baseline','modified');
ax.YGrid = 'on';
ylim([0 550]);
% xlim([0.015 0.06])
% ytickformat('percentage')
fontsize([lgd,line1,line2,xlab],22,"points");
% exportgraphics(gcf,'sim_l1.pdf','BackgroundColor','none','ContentType','vector')


%% factor separate plot
% figure
% for r=1:R
%     ax1(r)=subplot(3,R,r);
%     histogram(e_l1(:,r));
%     hold on
%     histogram(e_l2(:,r));
%     yL=get(gca,'YLim');
%     line([mean(e_l1(:,r)), mean(e_l1(:,r))], yL, 'LineWidth', 1, 'Color', 'r');
%     text(mean(e_l1(:,r)),yL(2)-20,num2str(round(mean(e_l1(:,r)),2,'significant')),'VerticalAlignment','top'); 
%     line([mean(e_l2(:,r)), mean(e_l2(:,r))], yL, 'LineWidth', 1, 'Color', 'r');
%     text(mean(e_l2(:,r)),yL(2),num2str(round(mean(e_l2(:,r)),2,'significant')),'VerticalAlignment','top','HorizontalAlignment','right');
%     title(['\lambda_',num2str(r)]);
%     ylim(yL);
%     legend('baseline','modified','location','southeast')
%     ax2(r)=subplot(3,R,r+R);
%     histogram(e_m1(:,r));
%     hold on
%     histogram(e_m2(:,r));
%     yL=get(gca,'YLim');
%     line([mean(e_m1(:,r)), mean(e_m1(:,r))], yL, 'LineWidth', 1, 'Color', 'r');
%     text(mean(e_m1(:,r)),yL(2)-20,num2str(round(mean(e_m1(:,r)),2,'significant')),'VerticalAlignment','top');
%     line([mean(e_m2(:,r)), mean(e_m2(:,r))], yL, 'LineWidth', 1, 'Color', 'r');
%     text(mean(e_m2(:,r)),yL(2),num2str(round(mean(e_m2(:,r)),2,'significant')),'VerticalAlignment','top','HorizontalAlignment','right');
%     title(['\mu_',num2str(r)]);
%     ylim(yL);
%     legend('baseline','modified','location','southeast')
%     ax3(r)=subplot(3,R,r+2*R);
%     histogram(e_f1(:,r));
%     hold on
%     histogram(e_f2(:,r));
%     yL=get(gca,'YLim');
%     line([mean(e_f1(:,r)), mean(e_f1(:,r))], yL, 'LineWidth', 1, 'Color', 'r');
%     text(mean(e_f1(:,r)),yL(2)-20,num2str(round(mean(e_f1(:,r)),2,'significant')),'VerticalAlignment','top');
%     line([mean(e_f2(:,r)), mean(e_f2(:,r))], yL, 'LineWidth', 1, 'Color', 'r');
%     text(mean(e_f2(:,r)),yL(2),num2str(round(mean(e_f2(:,r)),2,'significant')),'VerticalAlignment','top','HorizontalAlignment','right');
%     title(['f_',num2str(r)]);
%     ylim(yL);
%     legend('baseline','modified','location','southeast')
% end
% set(gcf, 'Position',  [500, 500, 600, 500])
% exportgraphics(gcf,'alpha.pdf','BackgroundColor','none','ContentType','vector')
