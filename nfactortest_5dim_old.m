%% random sample split

clear;clc;
rng(23,'twister');
R0 = 2; % true rank
T = 50;
N1 = 10;
N2 = 20;
N3 = 30;
N4 = 40;
K = 5000; %# of repetitions


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

% d2=[0:0.05:1]; %0.4 for normal, 0.5 for t
d2 = 0;
% power1 = NaN(length(d2),1);power2 = NaN(length(d2),1);power3 = NaN(length(d2),1);
p1 = NaN(K,length(d2));p2 = NaN(K,length(d2));p3 = NaN(K,length(d2));p4 = NaN(K,length(d2));p5 = NaN(K,length(d2));
for i=1:length(d2)
    disp(d2(i));

% signal strength
s = sqrt(N1*N2*N3*N4*T)*([2,d2(i)])';D=diag(s);

R = 1; %rank estimated



% d1 = NaN(K,1);d2 = NaN(K,1);d3 = NaN(K,1);
% Sk1 = NaN(K,1);Sk2 = NaN(K,1);Sk3 = NaN(K,1);Sk4 = NaN(K,1);Sk5 = NaN(K,1);
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

    % random sample splitting N1
    ind1 = datasample(1:N2*N3*N4*T,N2*N3*N4*T/2,'Replace',false);
    ind1 = sort(ind1);
    ind2 = setdiff(1:N2*N3*N4*T,ind1);

    Y1 = Y_1(:,ind1);
    [~,S] = eig(Y1*Y1');
    [s_1,~] = sort(diag(S),'descend');
    
    Y2 = Y_1(:,ind2);
    [~,S] = eig(Y2*Y2');
    [s_2,~] = sort(diag(S),'descend');

    % estimate error variance
    Y_hat = double(tensor(ktensor(sqrt(s1(1:R)),{V1_hat(:,1:R),V2_hat(:,1:R),V3_hat(:,1:R),V4_hat(:,1:R),F_hat(:,1:R)})));
    Y_e = Y - Y_hat;
    Y_e = reshape(Y_e,[N1,N2*N3*N4*T]);
    Y_e = Y_e(:,ind2);
%     Y_e = Y_e(:,ind1);
    
    sig_u_hat = std(Y_e,1,"all");
    
    mean_u = mean(Y_e,"all");

    gamma_4 = mean((Y_e - mean_u).^4,"all");
    
    var_Sk = 2*(gamma_4 - sig_u_hat^4);

    % statistic
    Sk = sum(s_1(R+1:end))/sqrt(N2*N3*N4*T*(N1-R)/2)-sig_u_hat^2*sqrt(N2*N3*N4*T*(N1-R)/2);
%     Sk(k) = sum(s31(R+1:end))/sqrt(N*J*(T-R)/2)-sum(s32(R+1:end))/sqrt(N*J*(T-R)/2);
    
%     c = 2*sig_u_hat^2*norminv(0.975); % critical value
    c = sqrt(var_Sk) * norminv(0.975);
    p1(k,i) = 2*(1-normcdf(abs(Sk/sqrt(var_Sk))));

    %%%%%%%%%%%%% random sample splitting N2
    ind1 = datasample(1:N1*N3*N4*T,N1*N3*N4*T/2,'Replace',false);
    ind1 = sort(ind1);
    ind2 = setdiff(1:N1*N3*N4*T,ind1);

    Y1 = Y_2(:,ind1);
    [~,S] = eig(Y1*Y1');
    [s_1,~] = sort(diag(S),'descend');
    
    Y2 = Y_2(:,ind2);
    [~,S] = eig(Y2*Y2');
    [s_2,~] = sort(diag(S),'descend');

    % estimate error variance
    Y_hat = double(tensor(ktensor(sqrt(s2(1:R)),{V1_hat(:,1:R),V2_hat(:,1:R),V3_hat(:,1:R),V4_hat(:,1:R),F_hat(:,1:R)})));
    Y_e = Y - Y_hat;
    Y_e = reshape(permute(Y_e,[2,1,3,4,5]),[N2,N1*N3*N4*T]);
    Y_e = Y_e(:,ind2);
%     Y_e = Y_e(:,ind1);
    
    sig_u_hat = std(Y_e,1,"all");
    
    mean_u = mean(Y_e,"all");

    gamma_4 = mean((Y_e - mean_u).^4,"all");
    
    var_Sk = 2*(gamma_4 - sig_u_hat^4);

    % statistic
    Sk = sum(s_1(R+1:end))/sqrt(N1*N3*N4*T*(N2-R)/2)-sig_u_hat^2*sqrt(N1*N3*N4*T*(N2-R)/2);
%     Sk(k) = sum(s31(R+1:end))/sqrt(N*J*(T-R)/2)-sum(s32(R+1:end))/sqrt(N*J*(T-R)/2);
    
%     c = 2*sig_u_hat^2*norminv(0.975); % critical value
    c = sqrt(var_Sk) * norminv(0.975);
    p2(k,i) = 2*(1-normcdf(abs(Sk/sqrt(var_Sk))));

    %%%%%%%%% random sample splitting N3
    ind1 = datasample(1:N1*N2*N4*T,N1*N2*N4*T/2,'Replace',false);
    ind1 = sort(ind1);
    ind2 = setdiff(1:N1*N2*N4*T,ind1);

    Y1 = Y_3(:,ind1);
    [~,S] = eig(Y1*Y1');
    [s_1,~] = sort(diag(S),'descend');
    
    Y2 = Y_3(:,ind2);
    [~,S] = eig(Y2*Y2');
    [s_2,~] = sort(diag(S),'descend');

    % estimate error variance
    Y_hat = double(tensor(ktensor(sqrt(s3(1:R)),{V1_hat(:,1:R),V2_hat(:,1:R),V3_hat(:,1:R),V4_hat(:,1:R),F_hat(:,1:R)})));
    Y_e = Y - Y_hat;
    Y_e = reshape(permute(Y_e,[3,1,2,4,5]),[N3,N1*N2*N4*T]);
    Y_e = Y_e(:,ind2);
%     Y_e = Y_e(:,ind1);
    
    sig_u_hat = std(Y_e,1,"all");
    
    mean_u = mean(Y_e,"all");

    gamma_4 = mean((Y_e - mean_u).^4,"all");
    
    var_Sk = 2*(gamma_4 - sig_u_hat^4);

    % statistic
    Sk = sum(s_1(R+1:end))/sqrt(N1*N2*N4*T*(N3-R)/2)-sig_u_hat^2*sqrt(N1*N2*N4*T*(N3-R)/2);
%     Sk(k) = sum(s31(R+1:end))/sqrt(N*J*(T-R)/2)-sum(s32(R+1:end))/sqrt(N*J*(T-R)/2);
    
%     c = 2*sig_u_hat^2*norminv(0.975); % critical value
    c = sqrt(var_Sk) * norminv(0.975);
    p3(k,i) = 2*(1-normcdf(abs(Sk/sqrt(var_Sk))));

    %%%%%%%%% random sample splitting N4
    ind1 = datasample(1:N1*N2*N3*T,N1*N2*N3*T/2,'Replace',false);
    ind1 = sort(ind1);
    ind2 = setdiff(1:N1*N2*N3*T,ind1);

    Y1 = Y_4(:,ind1);
    [~,S] = eig(Y1*Y1');
    [s_1,~] = sort(diag(S),'descend');
    
    Y2 = Y_4(:,ind2);
    [~,S] = eig(Y2*Y2');
    [s_2,~] = sort(diag(S),'descend');

    % estimate error variance
    Y_hat = double(tensor(ktensor(sqrt(s4(1:R)),{V1_hat(:,1:R),V2_hat(:,1:R),V3_hat(:,1:R),V4_hat(:,1:R),F_hat(:,1:R)})));
    Y_e = Y - Y_hat;
    Y_e = reshape(permute(Y_e,[4,1,2,3,5]),[N4,N1*N2*N3*T]);
    Y_e = Y_e(:,ind2);
%     Y_e = Y_e(:,ind1);
    
    sig_u_hat = std(Y_e,1,"all");
    
    mean_u = mean(Y_e,"all");

    gamma_4 = mean((Y_e - mean_u).^4,"all");
    
    var_Sk = 2*(gamma_4 - sig_u_hat^4);

    % statistic
    Sk = sum(s_1(R+1:end))/sqrt(N1*N2*N3*T*(N4-R)/2)-sig_u_hat^2*sqrt(N1*N2*N3*T*(N4-R)/2);
%     Sk(k) = sum(s31(R+1:end))/sqrt(N*J*(T-R)/2)-sum(s32(R+1:end))/sqrt(N*J*(T-R)/2);
    
%     c = 2*sig_u_hat^2*norminv(0.975); % critical value
    c = sqrt(var_Sk) * norminv(0.975);
    p4(k,i) = 2*(1-normcdf(abs(Sk/sqrt(var_Sk))));

     %%%%%%%%% random sample splitting T
    ind1 = datasample(1:N1*N2*N3*N4,N1*N2*N3*N4/2,'Replace',false);
    ind1 = sort(ind1);
    ind2 = setdiff(1:N1*N2*N3*N4,ind1);

    Y1 = Y_5(:,ind1);
    [~,S] = eig(Y1*Y1');
    [s_1,~] = sort(diag(S),'descend');
    
    Y2 = Y_5(:,ind2);
    [~,S] = eig(Y2*Y2');
    [s_2,~] = sort(diag(S),'descend');

    % estimate error variance
    Y_hat = double(tensor(ktensor(sqrt(s5(1:R)),{V1_hat(:,1:R),V2_hat(:,1:R),V3_hat(:,1:R),V4_hat(:,1:R),F_hat(:,1:R)})));
    Y_e = Y - Y_hat;
    Y_e = reshape(permute(Y_e,[5,1,2,3,4]),[T,N1*N2*N3*N4]);
    Y_e = Y_e(:,ind2);
%     Y_e = Y_e(:,ind1);
    
    sig_u_hat = std(Y_e,1,"all");
    
    mean_u = mean(Y_e,"all");

    gamma_4 = mean((Y_e - mean_u).^4,"all");
    
    var_Sk = 2*(gamma_4 - sig_u_hat^4);

    % statistic
    Sk = sum(s_1(R+1:end))/sqrt(N1*N2*N3*N4*(T-R)/2)-sig_u_hat^2*sqrt(N1*N2*N3*N4*(T-R)/2);
%     Sk(k) = sum(s31(R+1:end))/sqrt(N*J*(T-R)/2)-sum(s32(R+1:end))/sqrt(N*J*(T-R)/2);
    
%     c = 2*sig_u_hat^2*norminv(0.975); % critical value
    c = sqrt(var_Sk) * norminv(0.975);
    p5(k,i) = 2*(1-normcdf(abs(Sk/sqrt(var_Sk))));

end

% power1(i) = sum(d1)/K;
% power2(i) = sum(d2)/K;
% power3(i) = sum(d3)/K;

end

% save('power3.mat','power')

p = NaN(K,length(d2),5);
p(:,:,1) = p1;p(:,:,2) = p2;p(:,:,3) = p3;p(:,:,4) = p4;p(:,:,5) = p5;
p_med = 2*median(p,3);
power_med = sum(p_med<0.05,1)./K;
p_max = max(p,[],3);
power_max = sum(p_max<0.05,1)./K;
p_min = min(p,[],3);
power_min = sum(p_min<0.05,1)./K;
p_mean = mean(p,3);
power_mean = sum(p_mean<0.05,1)./K;