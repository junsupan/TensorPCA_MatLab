% tensor decomposition via PCA, simulations
% tensor vs 2-way, model complexity argument
clear;clc;
rng(23,'twister');
alpha1 = 1;
alpha2 = 1;
alpha3 = 1;
R = 5;
T = 50;
N = 50;
J = 50;
K = 100; %# of repetitions

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

rmse_ts = NaN(K,R);rmse_tw = NaN(K,R);
R2_ts = NaN(K,R);R2_tw = NaN(K,R);
% err_ts = NaN(K,R);err_tw = NaN(K,R);
% abserr_ts = NaN(K,R);abserr_tw = NaN(K,R);

s = sqrt(N^alpha1*J^alpha2*T^alpha3)*(R:-1:1)';D=diag(s);
for k = 1:K
    if mod(k, 10) == 0
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
% recover factor strength
s0 = (s1+s2+s3)/3;
[s_ts,~] = findsigma(Y,{L_hat,M_hat,F_hat},s0);
% root mean squared error, idiosyncratic variation
for r=1:R
    Y_e = Y - double(tensor(ktensor(s_ts(1:r),{L_hat(:,1:r),M_hat(:,1:r),F_hat(:,1:r)})));
    rmse_ts(k,r) = rms(Y_e,'all');
    R2_ts(k,r) = 1 - var(Y_e,0,'all')/var(Y,0,'all');
end

% beta
% [Gamma,S] = eig(Y_3'*Y_3);
% [s4,ind] = sort(diag(S),'descend');
% beta_hat = Gamma(:,ind(1:R));
% s4 = sqrt(s4(1:R));
% s0 = (s3+s4)/2;
% [s_tw,Y_3_e] = findsigma(Y_3,{F_hat,beta_hat},s0);

% SVD
[F_tw,S,Beta_tw] = svd(Y_3);
F_tw = F_tw(:,1:R);
Beta_tw = Beta_tw(:,1:R);
s_tw = diag(S);s_tw = s_tw(1:R);
for r=1:R
    Y_3_e = Y_3 - F_tw(:,1:r)*diag(s_tw(1:r))*Beta_tw(:,1:r)';
    rmse_tw(k,r) = rms(Y_3_e,'all');
    R2_tw(k,r) = 1 - var(Y_3_e,0,'all')/var(Y,0,'all');
end

% err_ts(k,:) = s-abs(s_ts);
% err_tw(k,:) = s-abs(s_tw);
% abserr_ts(k,:) = abs(s-abs(s_ts));
% abserr_tw(k,:) = abs(s-abs(s_tw));

end

cplxity_ts = NaN(1,R);cplxity_tw = NaN(1,R);
npara_ts = NaN(1,R);npara_tw = NaN(1,R);
for r=1:R
    cplxity_ts(r) = (N+J+T)*r/(N*J*T);
    cplxity_tw(r) = (N*J+T)*r/(N*J*T);
    npara_ts(r) = (N+J+T)*r;
    npara_tw(r) = (N*J+T)*r;
end
R2_ts_avg = mean(R2_ts,1);
R2_ts_std = std(R2_ts,0,1);
R2_tw_avg = mean(R2_tw,1);
R2_tw_std = std(R2_tw,0,1);

%%
figure;
plot(npara_ts,R2_ts_avg,'-s','LineWidth',2, ...
    'MarkerSize',10,...
    'Color',[0 0.4470 0.7410],...
    'MarkerFaceColor',[0 0.4470 0.7410],...
    'MarkerEdgeColor','k');
hold on;
plot(npara_tw,R2_tw_avg,'-s','LineWidth',2, ...
    'MarkerSize',10,...
    'Color',[0.8500 0.3250 0.0980],...
    'MarkerFaceColor',[0.8500 0.3250 0.0980],...
    'MarkerEdgeColor','k');
ylabel('R^2');
xlabel('Number of Parameters')

grid on;
% yL = [0.2 1.02];
yL = [0.4 1];
ylim(yL);
% xlim([-5000 105000]);
% line([npara_tw(1), npara_tw(1)], yL, 'LineWidth', 1, 'Color', [0.4660 0.6740 0.1880],'LineStyle','--');
% line([npara_tw(2), npara_tw(2)], yL, 'LineWidth', 1, 'Color', [0.4660 0.6740 0.1880],'LineStyle','--');
legend('3-way','2-way','','','Location','southeast');
set(gcf, 'Position',  [500, 500, 600, 400]);
set(gca,'XMinorTick','on','YMinorTick','on');
for r=1:R
    text(npara_ts(r),R2_ts_avg(r)-0.03,num2str(r));
    text(npara_tw(r),R2_tw_avg(r)-0.03,num2str(r));
end
% for r=7:R
%     text(npara_tw(r),R2_tw_avg(r)-0.03,num2str(r));
% end
% text(npara_ts(8)+1500,R2_ts_avg(8),'\leftarrow 7,...,10');
% exportgraphics(gcf,'sim10factors.pdf','BackgroundColor','none','ContentType','vector')

% figure;
% b = bar([1:R],[R2_ts_avg',R2_tw_avg']);
% legend('R^2 ts','R^2 tw','Location','northwest');
% ylabel('R^2');
% xlabel('number of factors')

% figure;
% histogram(idiovar_ts(:,2));
% hold on
% histogram(idiovar_tw(:,2));
% legend('ts','tw');
% 
% figure;
% histogram(rmse_ts);
% hold on
% histogram(rmse_tw);
% legend('ts','tw');

% figure;
% histogram(err_ts(:,1));
% hold on
% histogram(err_tw(:,1));
% legend('ts','tw');
% rms(err_ts,1)
% rms(err_tw,1)
% quantile(idiovar_ts,0.025)
