function [s_hat, Y_e] = findsigma2(Y,est,s0)
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
    options = optimoptions('fminunc','display','none');
    fun = @(s)sum((Y - double(tensor(ktensor(s,est)))).^2,'all');
    [s_hat,~] = fminunc(fun,s0,options);
    Y_e = Y - double(tensor(ktensor(s_hat,est)));
end

