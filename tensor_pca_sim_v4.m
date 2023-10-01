% tensor decomposition via PCA, simulations
% tensor PCA vs ALS
clear;clc;
rng(23,'twister');
alpha1 = 1;
alpha2 = 1;
alpha3 = 1;
R = 4;
T = 100;
N = 30;
J = 20;
K = 5000; %# of repetitions

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
s = sqrt(N^alpha1*J^alpha2*T^alpha3)*(R:-1:1)';D=diag(s);

R = 1; %rank estimated
err_l_ts = NaN(K,R);err_m_ts = NaN(K,R);err_f_ts = NaN(K,R);
err_l_als= NaN(K,R);err_m_als = NaN(K,R);err_f_als = NaN(K,R);
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
s1 = sqrt(s1(1:R));
% Mu
[Gamma,S] = eig(Y_2*Y_2');
[s2,ind] = sort(diag(S),'descend');
M_hat = Gamma(:,ind(1:R));
s2 = sqrt(s2(1:R));
% F
[Gamma,S] = eig(Y_3*Y_3');
[s3,ind] = sort(diag(S),'descend');
F_hat = Gamma(:,ind(1:R));
s3 = sqrt(s3(1:R));

for r=1:R
    err_l_ts(k,r) = norm(L(:,r)-L_hat(:,r)*sign(L_hat(:,r)'*L(:,r)));
    err_m_ts(k,r) = norm(M(:,r)-M_hat(:,r)*sign(M_hat(:,r)'*M(:,r)));
    err_f_ts(k,r) = norm(F(:,r)-F_hat(:,r)*sign(F_hat(:,r)'*F(:,r)));
end

% % recover factor strength
% s0 = (s1+s2+s3)/3;
% [s_ts,~] = findsigma(Y,{L_hat,M_hat,F_hat},s0);
% % root mean squared error, idiosyncratic variation
% for r=1:R
%     Y_e = Y - double(tensor(ktensor(s_ts(1:r),{L_hat(:,1:r),M_hat(:,1:r),F_hat(:,1:r)})));
%     rmse_ts(k,r) = rms(Y_e,'all');
%     R2_ts(k,r) = 1 - var(Y_e,0,'all')/var(Y,0,'all');
% end

% ALS
CPALS = cp_als(tensor(Y),R,'printitn',0);
L_als = CPALS.U{1};
M_als = CPALS.U{2};
F_als = CPALS.U{3};

for r=1:R
    err_l_als(k,r) = norm(L(:,r)-L_als(:,r)*sign(L_als(:,r)'*L(:,r)));
    err_m_als(k,r) = norm(M(:,r)-M_als(:,r)*sign(M_als(:,r)'*M(:,r)));
    err_f_als(k,r) = norm(F(:,r)-F_als(:,r)*sign(F_als(:,r)'*F(:,r)));
end

end


%%
r = 1; %plot r-th factor
err_ts = err_f_ts(:,r);err_als = err_f_als(:,r);
figure
t = tiledlayout(1,2,'TileSpacing','compact');
bgAx = axes(t,'XTick',[],'YTick',[],'Box','off');
bgAx.Layout.TileSpan = [1 2];
ax1 = axes(t);
ind = find(err_als<0.8);
a = histogram(ax1,err_als(ind));
hold on
histogram(ax1,err_ts)
uistack(a,'top')
ax1.YGrid = 'on';
% xline(ax1,max([err_ts;err_als(ind)]),'-.');
% ax1.Box = 'off';
legend('tensorPCA','tensorALS','Location','northwest')
% xlim(ax1,[0 15])
% Create second plot
ax2 = axes(t);
ax2.Layout.Tile = 2;
ind = find(err_als>=1.41);
histogram(ax2,err_als(ind))
% xline(ax2,min(err_als(ind)),'-.');
% ax2.YAxis.Visible = 'off';
% ax2.Box = 'off';
set(ax2,'YTickLabel',[])
ax2.YGrid = 'on';
% xlim(ax2,[45 60])

% Link the axes
linkaxes([ax1 ax2], 'y')
xlabel(t,'Estimation Error')
ylabel(t,'Count')

% exportgraphics(gcf,'vsals_l1.pdf','BackgroundColor','none','ContentType','vector')
