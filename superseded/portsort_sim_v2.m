% tensor decomposition via PCA, simulations
clear;
rng(23,'twister');
alpha1 = 1;
alpha2 = 1;
alpha3 = 1;
R = 1;
K = 5000; %# of repetitions

T = 60;
N = 100;
J = [10,10];

sizes=2;
l_l = NaN(K,R,sizes);l_m = NaN(K,R,sizes);l_f = NaN(K,R,sizes);
l_l_hml = NaN(K,R,sizes);l_f_hml = NaN(K,R,sizes);
R2_ts = NaN(K,R,sizes);R2_hml = NaN(K,R,sizes);
for d=1:2

% generate Lambda
L = rand(N,R)-0.5;
if R>1
    for r=2:R
        for s=1:r-1
            L(:,r) = L(:,r)- L(:,r)'*L(:,s)*L(:,s)/(L(:,s)'*L(:,s));
        end
    end
end
L_norm = L/(sqrtm(L'*L));
% L_norm = get_orthonormal(N,R);
L = L_norm*sqrt(N^alpha1);

% generate Mu
M = NaN(J(d),R);
M(:,1) = 1:J(d);
% M(:,2) = ones(J(d),1);
% if R>1
%     for r=2:R
%         for s=1:r-1
%             M(:,r) = M(:,r)- M(:,r)'*M(:,s)*M(:,s)/(M(:,s)'*M(:,s));
%         end
%     end
% end
M_norm = M/(sqrtm(M'*M));
% M_norm = get_orthonormal(J(d),R);
M = M_norm*sqrt(J(d)^alpha1);

% generate F
rho = [1];
sig_e = 0.1;
F = NaN(T,R);
F = ones(T,1)*rho + sig_e*randn(T,R);
if R>1
    for r=2:R
        for s=1:r-1
            F(:,r) = F(:,r)- F(:,r)'*F(:,s)*F(:,s)/(F(:,s)'*F(:,s));
        end
    end
end
F_norm = F/(sqrtm(F'*F));
% F_norm = F_norm(:,[2,1]); % variance of f2 significantly smaller than f1
F = F_norm*sqrt(T^alpha3);

for k = 1:K
    if mod(k, 100) == 0
        disp(k);
    end

% generate tensor Y   https://www.tensortoolbox.org/ktensor_doc.html#7
sig_u = 1;sig_u_tail = [1,2];
s = [1];D=diag(s);
% s = [2*sqrt((N*J(d)*T)^a1);*sqrt((N*J(d)*T)^a2)];
Y = ktensor(s,{L,M,F});
Y = tensor(Y);
U = sig_u*randn([N,J(d),T]);
for t=1:T
    U(:,1,t) = sig_u_tail(d)*randn(N,1);
    U(:,J(d),t) = sig_u_tail(d)*randn(N,1);
end
Y = double(Y) + U; Y_ts = tensor(Y);
Y_1 = double(reshape(Y_ts,[N,J(d)*T]));
Y_2 = double(reshape(permute(Y_ts,[2,1,3]),[J(d),N*T]));
Y_3 = double(reshape(permute(Y_ts,[3,1,2]),[T,N*J(d)]));
Y_hml = NaN(N,T);
for i=1:N
    Y_hml(i,:) = Y(i,J(d),:) - Y(i,1,:);
end

% tensor decomposition via PCA 
% Lambda
[Gamma_1,S_1] = eig(Y_1*Y_1');
% Gamma_1 = Gamma_1*sqrt(N);
[s_1,ind] = sort(diag(S_1),'descend');
L_hat = Gamma_1(:,ind(1:R));
% Mu
[Gamma_2,S_2] = eig(Y_2*Y_2');
% Gamma_2 = Gamma_2*sqrt(J(d));
[s_2,ind] = sort(diag(S_2),'descend');
M_hat = Gamma_2(:,ind(1:R));
% F
[Gamma_3,S_3] = eig(Y_3*Y_3');
% Gamma_3 = Gamma_3*sqrt(T);
[s_3,ind] = sort(diag(S_3),'descend');
F_hat = Gamma_3(:,ind(1:R));
% calculate estimation error
for r=1:R
    l_l(k,r,d) = norm(L_norm(:,r)-L_hat(:,r)*sign(L_hat(:,r)'*L_norm(:,r)));
    l_m(k,r,d) = norm(M_norm(:,r)-M_hat(:,r)*sign(M_hat(:,r)'*M_norm(:,r)));
    l_f(k,r,d) = norm(F_norm(:,r)-F_hat(:,r)*sign(F_hat(:,r)'*F_norm(:,r)));
end

% PCA on HML sorted portfolios
% Lambda
[Gamma_1,S_1] = eig(Y_hml*Y_hml');
[s_1,ind] = sort(diag(S_1),'descend');
L_hat_hml = Gamma_1(:,ind(1:R));
% F
[Gamma_3,S_3] = eig(Y_hml'*Y_hml);
[s_3,ind] = sort(diag(S_3),'descend');
F_hat_hml = Gamma_3(:,ind(1:R));

for r=1:R
    l_l_hml(k,r,d) = norm(L_norm(:,r)-L_hat_hml(:,r)*sign(L_hat_hml(:,r)'*L_norm(:,r)));
    l_f_hml(k,r,d) = norm(F_norm(:,r)-F_hat_hml(:,r)*sign(F_hat_hml(:,r)'*F_norm(:,r)));
end

% R2
for r=1:R
    [~,~,~,~,stats] = regress(F_hat(:,r),[ones(T,1),F(:,r)]);
    R2_ts(k,r,d) = stats(1);
    [~,~,~,~,stats] = regress(F_hat_hml(:,r),[ones(T,1),F(:,r)]);
    R2_hml(k,r,d) = stats(1);
%     mdl = fitlm(F(:,r),F_hat(:,r));
%     R2_ts(k,r,d) = mdl.Rsquared.Ordinary;
%     mdl = fitlm(F(:,r),F_hat_hml(:,r));
%     R2_hml(k,r,d) = mdl.Rsquared.Ordinary;
end

end

end

% histogram(R2_ts);
% hold on;
% histogram(R2_hml);
% legend('tensor','hml','location','southeast');

% plot(F_norm)
% hold on
% plot(F_hat)
% plot(F_hat_hml)
% legend('true','tensor','hml')
%% factor separate plot
% d=1;
% fig1_l = l_l(:,:,d); fig2_l = l_l_hml(:,:,d);%exp 1
% fig1_f = l_f(:,:,d); fig2_f = l_f_hml(:,:,d);
fig1_l = l_l(:,:,2); fig2_l = l_l(:,:,1);%exp 2
fig1_f = l_f(:,:,2); fig2_f = l_f(:,:,1);
figure;
for r=1:R
    ax1(r)=subplot(R,2,2*r-1);
    histogram(fig1_l(:,r));
    hold on
    histogram(fig2_l(:,r));
    yL=get(gca,'YLim');
    line([mean(fig1_l(:,r)), mean(fig1_l(:,r))], yL, 'LineWidth', 1, 'Color', 'r');
    text(mean(fig1_l(:,r)),yL(2)-30,['\leftarrow',num2str(round(mean(fig1_l(:,r)),2,'significant'))],'VerticalAlignment','top'); 
    line([mean(fig2_l(:,r)), mean(fig2_l(:,r))], yL, 'LineWidth', 1, 'Color', 'r');
    text(mean(fig2_l(:,r)),yL(2),['\leftarrow',num2str(round(mean(fig2_l(:,r)),2,'significant'))],'VerticalAlignment','top');
    title(['\lambda']);
    ylim(yL);
    xlim([min(fig1_l(:,r))-std(fig1_l(:,r)),max(fig2_l(:,r))]);
    legend('decile','quintile','location','southeast');

    ax2(r)=subplot(R,2,2*r);
    histogram(fig1_f(:,r));
    hold on
    histogram(fig2_f(:,r));
    yL=get(gca,'YLim');
    line([mean(fig1_f(:,r)), mean(fig1_f(:,r))], yL, 'LineWidth', 1, 'Color', 'r');
    text(mean(fig1_f(:,r)),yL(2)-30,['\leftarrow',num2str(round(mean(fig1_f(:,r)),2,'significant'))],'VerticalAlignment','top');
    line([mean(fig2_f(:,r)), mean(fig2_f(:,r))], yL, 'LineWidth', 1, 'Color', 'r');
    text(mean(fig2_f(:,r)),yL(2),['\leftarrow',num2str(round(mean(fig2_f(:,r)),2,'significant'))],'VerticalAlignment','top');
    title(['f']);
    ylim(yL);
    xlim([min(fig1_f(:,r))-std(fig1_f(:,r)),max(fig2_f(:,r))])
    legend('decile','quintile','location','southeast');
end
set(gcf, 'Position',  [500, 500, 800, 200]);
pos = get( ax2(1), 'Position' );
pos(1)=0.5;pos(4)=0.8087;
set(ax2(1),'Position',pos);
% exportgraphics(gcf,'alpha.pdf','BackgroundColor','none','ContentType','vector')

%% R2 plot
d=2;
fig1_ts = R2_ts(:,:,d);fig1_hml = R2_hml(:,:,d);
for r=1:R
ax(r)=subplot(1,R,r);
histogram(fig1_ts(:,r));
hold on
histogram(fig1_hml(:,r));
yL=get(gca,'YLim');
line([mean(fig1_ts(:,r)), mean(fig1_ts(:,r))], yL, 'LineWidth', 1, 'Color', 'r');
text(mean(fig1_ts(:,r)),yL(2)-30,['\leftarrow',num2str(round(mean(fig1_ts(:,r)),2,'significant'))],'VerticalAlignment','top');
line([mean(fig1_hml(:,r)), mean(fig1_hml(:,r))], yL, 'LineWidth', 1, 'Color', 'r');
text(mean(fig1_hml(:,r)),yL(2),['\leftarrow',num2str(round(mean(fig1_hml(:,r)),2,'significant'))],'VerticalAlignment','top');
% title(['R^2']);
ylim(yL);
legend('tensor','hml','location','southeast');
end
set(gcf, 'Position',  [500, 500, 400, 200]);
