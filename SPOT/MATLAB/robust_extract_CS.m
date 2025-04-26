function [sol, output_info] = robust_extract_CS(Xs, mom_mat_rpt, total_var_num, eps)
    % robust minimizer extraction only with correlative sparsity
    % input: 
    %   Xs: cells of moment matrices, at least of length length(mom_mat_rpt)
    %   mom_mat_rpt: cells of moment matrices' one side rpts 
    %   total_var_num: number of polynomial variables
    %   eps: truncated tolerance for minimizer extraction
    % WARNING: each Xs{i} should be an "entire" moment matrix.
    % term sparsity is not supported here

    sol = zeros(total_var_num, 1);
    cnt = zeros(total_var_num, 1);
    mom_mat_num = length(mom_mat_rpt);

    xi_recovers = cell(mom_mat_num, 1);
    w_recovers = cell(mom_mat_num, 1);
    var_ids = cell(mom_mat_num, 1);

    for i = 1: mom_mat_num
        rpt = mom_mat_rpt{i};
        X = Xs{i};
        [mom_sub, Ks, var_id] = generate_Ks(X, rpt, total_var_num);
    
        input_info.mom_sub = mom_sub;
        input_info.Ks = Ks;
        input_info.eps = eps;
        [xi_recover, output_info_inner] = extraction_robust(input_info);
    
        [~, max_id] = max(output_info_inner.w_recover);
        sol(var_id) = sol(var_id) + xi_recover(:, max_id);
        cnt(var_id) = cnt(var_id) + 1;

        xi_recovers{i} = xi_recover;
        w_recovers{i} = output_info_inner.w_recover;
        var_ids{i} = var_id;
    end
    cnt(cnt == 0) = 1;
    sol = sol ./ cnt;

    output_info.xi_recovers = xi_recovers;
    output_info.w_recovers = w_recovers;
    output_info.var_ids = var_ids;
end


%% helper functions
function [mom_sub, Ks, var_id] = generate_Ks(X, rpt, total_var_num)
    % X should be an entire moment matrix
    s = size(rpt, 1);
    kappa = size(rpt, 2) / 2;
    n = total_var_num;
    rpt_short = rpt(:, kappa+1: end);
    M1 = repmat(rpt_short, s, 1);
    M2 = kron(rpt_short, ones(s, 1));
    Mv = [M1, M2];
    Mv = sort(Mv, 2);
    
    C = containers.Map('KeyType', 'uint64', 'ValueType', 'any');
    keys = get_key(Mv, n);
    for i = 1: s
        for j = 1: s
            id = (i-1) * s + j;
            C(keys{id}) = X(i, j);
        end
    end
    
    sub_size = 0;
    for i = 1: size(rpt, 1)
        if nnz(rpt(i, :)) == 1 || nnz(rpt(i, :)) == 0
            sub_size = sub_size + 1;
        end
    end
    mom_sub = X(1:sub_size, 1:sub_size);
    rpt_sub_short = rpt_short(1:sub_size, :);
    M1_sub = repmat(rpt_sub_short, sub_size, 1);
    M2_sub = kron(rpt_sub_short, ones(sub_size, 1));
    Mv_sub = [M1_sub, M2_sub];
    Mv_sub = sort(Mv_sub, 2);
    Mv_sub_short = Mv_sub(:, kappa+1: end);
    
    Ks = cell(sub_size-1, 1);
    for i = 1: sub_size-1
        rpt_short_single = rpt_sub_short(i+1, :);
        rpt_short_single = repmat(rpt_short_single, sub_size^2, 1);
        Kv = sort([rpt_short_single, Mv_sub_short], 2);
        id = get_key(Kv, n);
        K = zeros(size(mom_sub));
        for j1 = 1: sub_size
            for j2 = 1: sub_size
                tmp = (j1 - 1) * sub_size + j2;
                K(j1, j2) = C(id{tmp});
            end
        end
        Ks{i} = K;
    end

    var_id = rpt(2: sub_size, end);
end

function [xi_recover, output_info] = extraction_robust(input_info)
    n = length(input_info.Ks);

    % extract solutions: S, U
    [U, S] = sorteig(input_info.mom_sub);
    s = diag(S);
    if_truncate = false;
    for k = 1: length(s) - 1
        if s(k+1) / s(1) < input_info.eps
            if_truncate = true;
            break;
        end
    end
    if if_truncate
        S = S(1:k, 1:k);
        U = U(:, 1:k);
    end
    s = diag(S);
    S_sqrt = diag(sqrt(s));
    S_sqrt_inv = diag(1 ./ sqrt(s));
    
    % get localizing matrices K and YK
    Ks = input_info.Ks;
    YKs = cell(n, 1);
    for k = 1: n
        YK = S_sqrt_inv * U' * Ks{k} * U * S_sqrt_inv;
        YKs{k} = YK;
    end
    
    % extract solution
    YK_random = zeros(size(YKs{1}));
    for k = 1: n
        rand_num = 2 * rand() - 1;
        YK_random = YK_random + YKs{k} * rand_num;
    end
    YK_random = 0.5 * (YK_random + YK_random');
    [O, ~] = eig(YK_random);
    Ys = cell(size(S, 1), 1);
    for k = 1: n
        Y = O' * YKs{k} * O;
        Ys{k} = diag(Y);
    end
    xi_recover = zeros(n, size(S, 1));
    for k = 1: n
        xi_recover(k, :) = Ys{k};
    end
    tmp = O' * S_sqrt * U';
    v_recover = abs(tmp(:, 1)');
    output_info.w_recover = v_recover.^2;
    output_info.v_recover = v_recover;
end
