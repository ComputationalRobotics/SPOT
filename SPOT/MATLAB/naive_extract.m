% naive minimizer extraction for CSTSS
function [v_opt, output_info] = naive_extract(Xs, mom_mat_rpt, total_var_num)
    v_opt = zeros(total_var_num, 1);
    cnt = zeros(total_var_num, 1);
    mom_mat_num = length(mom_mat_rpt);
    ds = cell(mom_mat_num, 1);

    for i = 1: mom_mat_num
        rpt = mom_mat_rpt{i};
        idx = 0;
        for j = 1: size(rpt, 1)
            % monomials of degree 0 or 1
            if nnz(rpt(j, :)) == 1 || nnz(rpt(j, :)) == 0
                idx = idx + 1;
            else 
                break;
            end
        end
        if idx > 0
            X = Xs{i};
            X_small = X(1: idx, 1: idx);
            [V, D] = sorteig(X_small);
            vv = V(:, 1);
            vv = vv / vv(1);
            vv = vv(2:end);
            var_id = rpt(2:idx, end);
            v_opt(var_id) = v_opt(var_id) + vv;
            cnt(var_id) = cnt(var_id) + 1;

            ds{i} = diag(D);
        end
    end
    cnt(cnt == 0) = 1;
    v_opt = v_opt ./ cnt;

    output_info.ds = ds;
end