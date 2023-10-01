%two-way test Onatski's test
clear;clc;
rng(23,'twister');
R0 = 2;R=1;

K0 = 5; % parameter in TW distribution
T = 100;
N = 100;
K = 5000; %# of repetitions
d2 = 0:0.05:0.5;

% calculate critical value
% asydist = NaN(K,1);
% for k=1:K
%     if mod(k, 100) == 0
%         disp(k);
%     end
% %     Z = randn([min(T,N)-R,min(T,N)-R]);
% %     Z = tril(Z,-1);
% %     Z = Z + Z';
% %     Z(1:min(T,N)-R+1:end) = sqrt(2)*randn(min(T,N)-R,1);
%     
%     Z = randn([1000,1000]);
%     Z = tril(Z,-1);
%     Z = Z + Z';
%     Z(1:1001:end) = sqrt(2)*randn(1000,1);
% 
%     [~,s_z] = eig(Z);
%     s_z = sort(diag(s_z),'descend');
% %     asydist(m) = s_z(1)-s_z(end);
%     
%     asydist_i = NaN(K0-R,1);
%     for r=1:K0-R
%         asydist_i(r) = (s_z(r)-s_z(r+1))/(s_z(r+1)-s_z(r+2));
%     end
%     asydist(k) = max(asydist_i);
% end
% c = quantile(asydist,0.95);
c = 11.9853;

% generate Lambda
L = get_orthonormal(N,R0);
% L_norm = L/(sqrtm(L'*L));

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

% estimate and test
power = NaN(length(d2),1);
for i=1:length(d2)
disp(d2(i));

% signal strength
s = sqrt(N*T)*([2,d2(i)])';D=diag(s);

Sk = NaN(K,1);
for k = 1:K
%     if mod(m, 100) == 0
%         disp(m);
%     end
    % generate tensor Y   https://www.tensortoolbox.org/ktensor_doc.html#7
    sig_u = 1;
    Y = L*D*F';
    U = sig_u*randn([N,T]);
%     U = sig_u*(sqrt(1/2)*randn([N,T])+sqrt(1/2)*randn([N,T])*1i);
    Y = Y + U;

    [L_hat,S,F_hat] = svd(Y,"econ");
%     [L_eig,S_eig] = eig(Y*Y'); % identical to SVD

    L_hat = L_hat(:,1:R);
    s_hat = diag(S);
    gamma = s_hat.^2;
    s_hat = s_hat(1:R);
    F_hat = F_hat(:,1:R);

    % select correct sign
    for r=1:R
        L_hat(:,r) = L_hat(:,r)*sign(L_hat(:,r)'*L(:,r));
        F_hat(:,r) = F_hat(:,r)*sign(F_hat(:,r)'*F(:,r));
    end

    Y_hat = L_hat*diag(s_hat)*F_hat';
    Y_e = Y - Y_hat;

    sig_u_hat = std(Y_e,0,'all');

    Sk_i = NaN(K0-R,1);
    for r=1:K0-R
        Sk_i(r) = (gamma(R+r)-gamma(R+r+1))/(gamma(R+r+1)-gamma(R+r+2));
    end
    Sk(k) = max(Sk_i);

%     Sk(m) = (gamma(R+1) - gamma(min(T,N)))/(sig_u_hat^2*sqrt(max(T,N)));
%     Sk(k) = (gamma(R+1) - gamma(min(T,N)))/(sig_u^2*sqrt(max(T,N)));


end

% c = quantile(asydist,[0.025,0.975]);

% power(i) = sum(any([Sk<c(1),Sk>c(2)],2))/M;
power(i) = sum(Sk>c)/K;
end

% plotting Onatski statistics
% figure
% histogram(Sk2)
% title('Finite Sample Distribution')
% 
% figure
% histogram(asydist)
% title('Asymptotic Distribution')

%%
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

