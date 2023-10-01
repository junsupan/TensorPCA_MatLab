function [s_hat, Y_e] = findsigma(Y,est,s0)
    R = length(s0);
    N = size(Y);
    K = length(N);
    N_est = NaN(1,K);
    for k=1:K
        [N_est(k),~] = size(est{k});
    end
    % make sure Y and est are correctly aligned
    if any(N ~= N_est) && K == length(unique(N))
        ind = NaN(1,K);
        for k=1:K
            ind(k) = find(N_est==N(k));
        end
        est = est(ind);
    end
    % find sigma
    s_hat = zeros(R,1);
    Y_e = Y;
    options = optimoptions('fminunc','display','none');
    for r=1:R
        
        A = cell(1,K);
        for k=1:K
            A{k} = est{k}(:,r);
        end
        Y_r = double(tensor(ktensor(1,A)));
%         fun = @(s)rms(Y_e - s*Y_r,'all');
        fun = @(s)sum((Y_e - s*Y_r).^2,'all');
        [s_hat(r),~] = fminunc(fun,s0(r),options);

        Y_e = Y_e - s_hat(r)*Y_r;
    end
end

