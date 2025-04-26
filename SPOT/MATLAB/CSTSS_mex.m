function [result, res, coeff_info, aux_info] = CSTSS_mex(f, g, h, d, x, input_info)
    relax_mode = input_info.relax_mode; relax_mode = char(relax_mode);
    cs_mode = input_info.cs_mode; cs_mode = char(cs_mode);
    cs_cliques = input_info.cliques;
    ts_mode = input_info.ts_mode; ts_mode = char(ts_mode);
    ts_mom_mode = input_info.ts_mom_mode; ts_mom_mode = char(ts_mom_mode);
    ts_eq_mode = input_info.ts_eq_mode; ts_eq_mode = char(ts_eq_mode);
    if_solve = input_info.if_solve;
    kappa = d; aux_info.kappa = kappa;
    ts_cliques = [];

    %% parameter
    ticp = tic;
    n = size(x, 1); m_ineq = size(g, 1); m_eq = size(h, 1);
    dj_g = zeros(1, m_ineq); dj_h = zeros(1, m_eq);
    supp_rpt_g = cell(1, m_ineq); supp_rpt_h = cell(1, m_eq); [supp_rpt_f, coeff_f] = supp_rpt(f, x, 2 * d);
    coeff_g = cell(1, m_ineq); coeff_h = cell(1, m_eq); 
    for i = 1 : m_ineq
        dj_g(i) = ceil(deg(g(i)) / 2);
        [supp_rpt_g{i}, coeff_g{i}] = supp_rpt(g(i), x, 2 * d);
    end
    for i = 1 : m_eq
        dj_h(i) = deg(h(i)); 
        [supp_rpt_h{i}, coeff_h{i}] = supp_rpt(h(i), x, 2 * d);
    end
    time = toc(ticp);
    disp(['parameter time', num2str(time)]);

    aux_info.supp_rpt_f = supp_rpt_f;
    aux_info.supp_rpt_g = supp_rpt_g;
    aux_info.supp_rpt_h = supp_rpt_h;
    aux_info.coeff_f = full(coeff_f);
    aux_info.coeff_g = coeff_g;
    aux_info.coeff_h = coeff_h;
    aux_info.dj_g = dj_g;
    aux_info.dj_h = dj_h;

    %% initerface with C++
    tic_mex = tic;
    [cs_info, ts_info, moment_info, sos_info] = CSTSS_MATLAB_mex( ...
        kappa, n, ...
        full(coeff_f), supp_rpt_f, ... 
        coeff_g, supp_rpt_g, ...
        coeff_h, supp_rpt_h, ...
        dj_g, dj_h, ...
        relax_mode, ...
        cs_mode, ...
        ts_mode, ...
        ts_mom_mode, ...
        ts_eq_mode, ...
        cs_cliques, ...
        ts_cliques ...
    );
    toc_mex = toc(tic_mex);
    fprintf("Conversion time:  %6d\n", toc_mex);
    % cs_info.c_g = cs_info.c_g + 1;
    % cs_info.c_h = cs_info.c_h + 1;
    for i = 1: length(ts_info.tI)
        for j = 1: length(ts_info.tI{i})
            ts_info.tI{i}{j} = ts_info.tI{i}{j} + 1;
        end
    end
    moment_info.A_moment(:, 1:end-1) = moment_info.A_moment(:, 1:end-1) + 1;
    moment_info.C_moment(:, 1:end-1) = moment_info.C_moment(:, 1:end-1) + 1;
    sos_info.A_sos(:, 1:end-1) = sos_info.A_sos(:, 1:end-1) + 1;
    sos_info.a_sos(:, 1:end-1) = sos_info.a_sos(:, 1:end-1) + 1;
    
    %% solve via mosek
    if if_solve 
        if strcmp(relax_mode, 'MOMENT')
            [res, mosek_time] = mosek_standard_sdp_test_1(full(moment_info.A_moment), full(moment_info.C_moment), moment_info.b_moment, ts_info.tI_size);
            coeff_info.A = moment_info.A_moment;
            coeff_info.C = moment_info.C_moment;
            coeff_info.b = moment_info.b_moment;
        end
        if strcmp(relax_mode, 'SOS')
            [res, mosek_time, iter, pfeas, dfeas, max_residual] = mosek_standard_sdp_test_2(full(sos_info.A_sos), full(sos_info.a_sos), sos_info.b_sos, sos_info.c_sos, ts_info.tI_size);
            coeff_info.A = sos_info.A_sos;
            coeff_info.prob_a = sos_info.a_sos;
            coeff_info.b = sos_info.b_sos;
            coeff_info.prob_c = sos_info.c_sos;
            aux_info.iter = iter;
            aux_info.pfeas = pfeas;
            aux_info.dfeas = dfeas;
            aux_info.max_residual = max_residual;
        end
        aux_info.mosek_time = mosek_time;
        result = res.sol.itr.pobjval;
    else 
        result = [];
        res = [];
        coeff_info = [];
    end

    %% collect aux_info
    aux_info.cliques = cs_info.cI;
    aux_info.clique_size = (ts_info.tI_size)';
    aux_info.ts_info = ts_info.tI;
    aux_info.mon_rpt = cs_info.mon;
    aux_info.mon_rpt_g = cs_info.mon_g;
    aux_info.mon_rpt_h = cs_info.mon_h;
    aux_info.time = toc_mex;
end
