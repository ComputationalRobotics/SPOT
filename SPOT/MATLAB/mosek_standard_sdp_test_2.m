function [res, mosek_time, iter, pfeas, dfeas, max_residual] = mosek_standard_sdp_test_2(A, prob_a, b, prob_c, s)
    prob.blc = b;
    prob.buc = b;
    prob.bardim = s';

    % rho
    prob.c = prob_c;
    prob.a = sparse(prob_a(:, 1), prob_a(:, 2), prob_a(:, 3), length(b), length(prob_c));

    % bara
    prob.bara.subi = A(:, 1);
    prob.bara.subj = A(:, 2);
    prob.bara.subk = A(:, 4);
    prob.bara.subl = A(:, 3);
    prob.bara.val = A(:, 5);

    % solve
    tic;
    param.MSK_IPAR_INTPNT_MAX_ITERATIONS = 100;
    % param.MSK_DPAR_OPTIMIZER_MAX_TIME = 1800.0;
    
    try
        [~, res] = mosekopt('maximize info', prob, param);
        iter = res.info.MSK_IINF_INTPNT_ITER;
        pfeas = res.info.MSK_DINF_INTPNT_PRIMAL_FEAS;
        dfeas = res.info.MSK_DINF_INTPNT_DUAL_FEAS;
        max_residual = get_mosek_gap(res);
    catch
        iter = -1;
        pfeas = -1;
        dfeas = -1;
        max_residual = -1;
    end
    
    mosek_time = toc;
end

function gap = get_mosek_gap(res)
    dual_gap = abs(res.info.MSK_DINF_INTPNT_DUAL_FEAS);
    primal_gap = abs(res.info.MSK_DINF_INTPNT_PRIMAL_FEAS);
    dual_obj_val = res.sol.itr.dobjval;
    primal_obj_val = res.sol.itr.pobjval;

    dp_gap = abs(primal_obj_val - dual_obj_val) / (1 + abs(primal_obj_val) + abs(dual_obj_val));
    gap = max([dual_gap, primal_gap, dp_gap]);
end