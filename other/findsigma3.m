function [s_hat, Y_e, est_new] = findsigma3(Y,est,s0)
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
    st1 = 1:R;st2 = 1:R;st3 = 1:R;
    est_new = est;
    for r=1:R
        set = cartprod(st1,st2,st3);
        fval0 = inf;
        for i=1:size(set,1)
            A = cell(1,3);
            A{1} = est{1}(:,set(i,1));
            A{2} = est{2}(:,set(i,2));
            A{3} = est{3}(:,set(i,3));
            Y_r0 = double(tensor(ktensor(1,A)));
            %         fun = @(s)rms(Y_e - s*Y_r,'all');
            fun = @(s)sum((Y_e - s*Y_r0).^2,'all');
            [si,fval] = fminunc(fun,s0(r),options);
            if fval<fval0
                fval0 = fval;
                s_hat(r) = si;
                est_new{1}(:,r) = A{1};
                est_new{2}(:,r) = A{2};
                est_new{3}(:,r) = A{3};
                set0 = set(i,:);
                Y_r = Y_r0;
            end
        end
        st1 = setdiff(st1,set0(1));
        st2 = setdiff(st1,set0(2));
        st3 = setdiff(st1,set0(3));

        Y_e = Y_e - s_hat(r)*Y_r;
    end
end

