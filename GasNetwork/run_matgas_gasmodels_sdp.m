function sdp = run_matgas_gasmodels_sdp(nlp)
%RUN_MATGAS_GASMODELS_SDP Moment/SOS relaxation of GasModels-compatible LS.
%
%   sdp = run_matgas_gasmodels_sdp(nlp)
%
% Input nlp is the output assembled by run_matgas_gasmodels_ls before its
% SDP field is attached.  The direct fmincon model remains in its stable
% q*abs(q) form.  This routine adds a_e=abs(q_e) only inside the polynomial
% optimization problem:
%
%   P_i - P_j - K*q_e*a_e = 0,
%   a_e^2 - q_e^2 = 0,  a_e >= 0.
%
% Thus the SDP and fmincon feasible sets agree, while the singular lifting
% never appears in fmincon.  CSTSS uses g(z)>=0 and h(z)=0.
%
% Requires SPOT/msspoly, CSTSS_mex, and the CSTSS recovery helpers.

    local_validate_input(nlp);
    opts = nlp.options;
    sdp = struct('status', 'initializing');

    if exist('msspoly', 'class') ~= 8
        warning('run_matgas_gasmodels_sdp:MissingMsspoly', ...
            'msspoly/SPOT was not found. Skipping the SDP relaxation.');
        sdp.status = 'skipped_no_msspoly';
        return;
    end
    if exist('CSTSS_mex', 'file') == 0
        warning('run_matgas_gasmodels_sdp:MissingCSTSS', ...
            'CSTSS_mex was not found. Skipping the SDP relaxation.');
        sdp.status = 'skipped_no_cstss';
        return;
    end
    if exist('msspoly_clean', 'file') == 0
        warning('run_matgas_gasmodels_sdp:MissingClean', ...
            'msspoly_clean was not found. Skipping the SDP relaxation.');
        sdp.status = 'skipped_no_msspoly_clean';
        return;
    end
    if exist('recover_mosek_sol_blk', 'file') == 0
        warning('run_matgas_gasmodels_sdp:MissingRecovery', ...
            'recover_mosek_sol_blk was not found. Skipping the SDP relaxation.');
        sdp.status = 'skipped_no_recovery';
        return;
    end

    model = local_build_sdp_model(nlp);
    [z, objective, inequality, equality] = local_build_pop(nlp, model);

    x_nlp_lifted = [nlp.x(:); ...
        abs(nlp.x(nlp.index.q(nlp.data.pipe_idx)))];
    check = local_numeric_pop(x_nlp_lifted, nlp, model);
    check.objective_difference = check.objective ...
        - nlp.objective_weighted_shed;
    check.feasible = check.max_inequality_violation <= opts.pop_feas_tol ...
        && check.max_equality_violation <= opts.pop_feas_tol;
    sdp.nlp_pop_check = check;

    if opts.verbose
        fprintf('\n=== Polynomial SDP model ===\n');
        fprintf('Variables (base / SDP-only qabs / total): %d / %d / %d\n', ...
            model.nbase, model.npipe, model.nsdp);
        fprintf('Constraints before cleaning (h / g)    : %d / %d\n', ...
            numel(equality), numel(inequality));
        fprintf('NLP point on POP: df=%+.3e  max(-g,0)=%.3e  max|h|=%.3e\n', ...
            check.objective_difference, check.max_inequality_violation, ...
            check.max_equality_violation);
    end
    if abs(check.objective_difference) > 10 * opts.pop_feas_tol || ...
            ~check.feasible
        warning('run_matgas_gasmodels_sdp:PopCheck', ...
            ['The fmincon solution does not satisfy the independently ', ...
             'constructed POP at tolerance %.3e.'], opts.pop_feas_tol);
    end

    [equality, ~] = msspoly_clean(equality, z, 1e-14, true);
    [inequality, ~] = msspoly_clean(inequality, z, 1e-14, true);
    [objective, objective_scale_factor] = ...
        msspoly_clean(objective, z, 1e-14, true);

    if opts.report_pop_coeff_ranges
        sdp.pop_coeff = local_pop_coefficient_ranges( ...
            z, objective, inequality, equality, opts.kappa, opts.verbose);
    else
        sdp.pop_coeff = struct();
    end

    params = struct();
    params.if_mex = true;
    params.kappa = opts.kappa;
    params.relax_mode = 'SOS';
    params.cs_mode = char(opts.cs_mode);
    params.ts_mode = 'NON';
    params.ts_mom_mode = 'NON';
    params.ts_eq_mode = 'NON';
    params.if_solve = true;
    params.cliques = [];

    if opts.verbose
        fprintf('CSTSS: kappa=%d  cs_mode=%s\n', ...
            opts.kappa, char(opts.cs_mode));
    end

    [result, res, coeff_info, aux_info] = CSTSS_mex( ...
        objective, inequality, equality, opts.kappa, z, params); %#ok<ASGLU>

    aux_info.total_var_num = model.nsdp;
    aux_info.base_var_num = model.nbase;
    aux_info.sdp_qabs_var_num = model.npipe;
    [solver_status_known, solver_status_ok, solver_status_text] = ...
        local_mosek_solution_status(res);
    if solver_status_known && ~solver_status_ok
        warning('run_matgas_gasmodels_sdp:SolverStatus', ...
            ['CSTSS/MOSEK returned non-optimal solution status %s. ', ...
             'No SDP bound will be reported.'], solver_status_text);
        sdp.status = 'solver_not_optimal';
        sdp.solver_status = solver_status_text;
        sdp.result = result;
        sdp.res = res;
        sdp.relax_info = aux_info;
        return;
    end

    blk = cell(numel(aux_info.clique_size), 2);
    for k = 1:numel(aux_info.clique_size)
        blk{k, 1} = 's';
        blk{k, 2} = aux_info.clique_size(k);
    end
    [Xopt, ~, Sopt, mosek_objective] = recover_mosek_sol_blk(res, blk);
    if isempty(mosek_objective) || ~isnumeric(mosek_objective) || ...
            ~isreal(mosek_objective) || ~isfinite(mosek_objective(1)) || ...
            ~(isscalar(objective_scale_factor) && ...
              isnumeric(objective_scale_factor) && ...
              isreal(objective_scale_factor) && ...
              isfinite(objective_scale_factor))
        warning('run_matgas_gasmodels_sdp:InvalidObjective', ...
            'CSTSS/MOSEK did not return a finite recoverable objective.');
        sdp.status = 'invalid_solver_objective';
        sdp.solver_status = solver_status_text;
        sdp.result = result;
        sdp.res = res;
        sdp.relax_info = aux_info;
        return;
    end
    if ~solver_status_known
        warning('run_matgas_gasmodels_sdp:UncheckedSolverStatus', ...
            ['MOSEK solution status was not exposed in the returned struct; ', ...
             'the finite recovered objective will be reported as unchecked.']);
    end
    if strcmp(params.relax_mode, 'MOMENT')
        Xs = Xopt;
    else
        Xs = Sopt;
        for k = 1:numel(Xs)
            Xs{k} = -Xs{k};
        end
    end

    [moment_rpt, can_extract] = local_moment_representations( ...
        aux_info, params.if_mex);
    robust = [];
    robust_info = [];
    if can_extract && exist('robust_extract_CS', 'file') ~= 0
        try
            [robust, robust_info] = robust_extract_CS( ...
                Xs, moment_rpt, model.nsdp, 1e-2);
        catch ME
            warning('run_matgas_gasmodels_sdp:RobustExtract', ...
                'Robust moment extraction failed: %s', ME.message);
        end
    end
    naive = [];
    naive_info = [];
    if can_extract && exist('naive_extract', 'file') ~= 0
        try
            [naive, naive_info] = naive_extract( ...
                Xs, moment_rpt, model.nsdp);
        catch ME
            warning('run_matgas_gasmodels_sdp:NaiveExtract', ...
                'Naive moment extraction failed: %s', ME.message);
        end
    end

    lower_bound = mosek_objective(1) * objective_scale_factor;
    upper_bound = nlp.objective_weighted_shed;
    gap = upper_bound - lower_bound;
    relative_gap = gap ...
        / max([1, abs(upper_bound), abs(lower_bound)]);
    priority = nlp.data.gasmodels_ls.dispatchable_delivery.priority;
    delivery_max = nlp.data.gasmodels_ls.dispatchable_delivery.maximum;
    weighted_target = priority.' * delivery_max;

    sdp.status = 'solved';
    sdp.solver_status_checked = solver_status_known;
    sdp.solver_status = solver_status_text;
    sdp.lower_bound = lower_bound;
    sdp.nlp_upper_bound = upper_bound;
    sdp.absolute_gap = gap;
    sdp.relative_gap = relative_gap;
    sdp.bound_order_valid = gap >= -max(opts.pop_feas_tol, 1e-7);
    sdp.weighted_served_upper_bound = ...
        min(weighted_target, weighted_target - lower_bound);
    sdp.weighted_served_nlp = weighted_target - upper_bound;
    if all(abs(priority - 1) <= 1e-12)
        unweighted_target = sum(delivery_max);
        sdp.dispatchable_shed_lower_bound_kgps = lower_bound;
        sdp.dispatchable_served_upper_bound_kgps = ...
            min(unweighted_target, unweighted_target - lower_bound);
    else
        sdp.dispatchable_shed_lower_bound_kgps = NaN;
        sdp.dispatchable_served_upper_bound_kgps = NaN;
    end
    sdp.objective_scale_factor = objective_scale_factor;
    sdp.result = result;
    sdp.res = res;
    sdp.Xs = Xs;
    sdp.relax_info = aux_info;
    sdp.variable_scale = model.scale;
    sdp.base_var_num = model.nbase;
    sdp.sdp_qabs_index = model.i_qabs;
    sdp.v_opt_naive = naive;
    sdp.output_info_naive = naive_info;
    sdp.zopt = robust(:);
    sdp.output_info_robust = robust_info;

    if ~isempty(robust) && numel(robust) == model.nsdp
        x_extract_lifted = model.scale .* robust(:);
        x_extract_base = x_extract_lifted(1:model.nbase);
        x_extract_direct = [x_extract_base; ...
            abs(x_extract_base(nlp.index.q(nlp.data.pipe_idx)))];
        sdp.x_extract_lifted = x_extract_lifted;
        sdp.x_extract_base = x_extract_base;
        sdp.extract_lifted_pop = local_numeric_pop( ...
            x_extract_lifted, nlp, model);
        sdp.extract_direct_pop = local_numeric_pop( ...
            x_extract_direct, nlp, model);
    else
        if ~isempty(robust)
            warning('run_matgas_gasmodels_sdp:ExtractLength', ...
                'Robust extract length %d does not match SDP variable count %d.', ...
                numel(robust), model.nsdp);
        end
        sdp.x_extract_lifted = [];
        sdp.x_extract_base = [];
        sdp.extract_lifted_pop = [];
        sdp.extract_direct_pop = [];
    end

    if lower_bound > upper_bound + max(opts.pop_feas_tol, 1e-7)
        warning('run_matgas_gasmodels_sdp:BoundOrder', ...
            ['SDP lower bound %.9g exceeds feasible fmincon value %.9g. ', ...
             'Check CSTSS status, relaxation/scaling, and POP diagnostics.'], ...
            lower_bound, upper_bound);
    end

    if opts.verbose
        fprintf('\n=== SDP relaxation result ===\n');
        fprintf('fmincon weighted shed (upper bound): %.9g\n', upper_bound);
        fprintf('SDP weighted shed (lower bound)     : %.9g\n', lower_bound);
        fprintf('Absolute / relative gap             : %.9g / %.3e\n', ...
            gap, relative_gap);
        fprintf('NLP weighted served                 : %.9g\n', ...
            sdp.weighted_served_nlp);
        fprintf('SDP upper bound on weighted served  : %.9g\n', ...
            sdp.weighted_served_upper_bound);
        if ~isempty(sdp.x_extract_base)
            fprintf('Robust extract direct max violation : %.3e\n', ...
                sdp.extract_direct_pop.max_violation);
        end
    end
end

% ======================================================================
% POP construction
% ======================================================================

function model = local_build_sdp_model(nlp)
    data = nlp.data;
    idx = nlp.index;
    matrices = nlp.model_matrices;
    model = struct();
    model.nbase = idx.nx;
    model.npipe = numel(data.pipe_idx);
    model.i_qabs = model.nbase + (1:model.npipe);
    model.nsdp = model.nbase + model.npipe;

    qcap = max(abs([data.qmin(data.pipe_idx), ...
        data.qmax(data.pipe_idx)]), [], 2);
    qabs_scale = max(qcap, ones(size(qcap)));
    model.lb = [matrices.lb(:); zeros(model.npipe, 1)];
    model.ub = [matrices.ub(:); qcap(:)];
    model.scale = [matrices.variable_scale(:); qabs_scale(:)];
    model.lb_hat = model.lb ./ model.scale;
    model.ub_hat = model.ub ./ model.scale;

    Aeq_hat = matrices.Aeq * spdiags( ...
        matrices.variable_scale(:), 0, model.nbase, model.nbase);
    model.balance_scale = max([full(sum(abs(Aeq_hat), 2)), ...
        abs(matrices.beq(:)), ones(data.N, 1)], [], 2);
end

function [z, objective, inequality, equality] = local_build_pop(nlp, model)
    data = nlp.data;
    idx = nlp.index;
    opts = nlp.options;
    matrices = nlp.model_matrices;

    z = msspoly('z', model.nsdp);
    x = z;
    for v = 1:model.nsdp
        x(v) = model.scale(v) * z(v);
    end
    xb = x(1:model.nbase);
    p2 = xb(idx.p2);
    q = xb(idx.q);
    sp = model.scale(idx.p2);
    sq = model.scale(idx.q);

    priority = data.gasmodels_ls.dispatchable_delivery.priority;
    objective = priority.' * xb(idx.shed);

    equality = spdiags(1 ./ model.balance_scale, 0, data.N, data.N) ...
        * (matrices.Aeq * xb - matrices.beq);

    for k = 1:model.npipe
        e = data.pipe_idx(k);
        i = data.from(e);
        j = data.to(e);
        a = x(model.i_qabs(k));
        sa = model.scale(model.i_qabs(k));
        Kbar = data.res_over_pa2(e);
        sw = max([sp(i), sp(j), abs(Kbar) * sq(e) * sa, 1]);
        sab = max([sa^2, sq(e)^2, 1]);
        equality = [equality; ...
            (p2(i) - p2(j) - Kbar * q(e) * a) / sw; ...
            (a^2 - q(e)^2) / sab]; %#ok<AGROW>
    end
    for k = 1:numel(data.short_idx)
        e = data.short_idx(k);
        i = data.from(e);
        j = data.to(e);
        ss = max([sp(i), sp(j), 1]);
        equality = [equality; (p2(i) - p2(j)) / ss]; %#ok<AGROW>
    end

    inequality = [z - model.lb_hat; model.ub_hat - z];

    if opts.sdp_strengthen
        for k = 1:model.npipe
            e = data.pipe_idx(k);
            i = data.from(e);
            j = data.to(e);
            a = x(model.i_qabs(k));
            sa = model.scale(model.i_qabs(k));
            sf = max([sa, sq(e), 1]);
            sd = max(sq(e) * max([sp(i), sp(j)]), 1);
            inequality = [inequality; ...
                (a - q(e)) / sf; ...
                (a + q(e)) / sf; ...
                q(e) * (p2(i) - p2(j)) / sd]; %#ok<AGROW>
        end
    end

    gm = data.gasmodels_ls;
    for k = 1:numel(gm.compressor_arc_index)
        e = gm.compressor_arc_index(k);
        i = data.from(e);
        j = data.to(e);
        rmin2 = data.crmin(e)^2;
        rmax2 = data.crmax(e)^2;
        direction = gm.compressor_directionality(e);
        if direction == 0
            span2 = rmax2 / rmin2;
            s1 = max([span2 * sp(i), sp(j), 1]);
            s2 = max([span2 * sp(j), sp(i), 1]);
            sd = max(sq(e) * max([sp(i), sp(j)]), 1);
            inequality = [inequality; ...
                (span2 * p2(i) - p2(j)) / s1; ...
                (span2 * p2(j) - p2(i)) / s2; ...
                -q(e) * (p2(i) - p2(j)) / sd]; %#ok<AGROW>
        elseif direction == 1
            s1 = max([rmax2 * sp(i), sp(j), 1]);
            s2 = max([sp(j), rmin2 * sp(i), 1]);
            inequality = [inequality; ...
                (rmax2 * p2(i) - p2(j)) / s1; ...
                (p2(j) - rmin2 * p2(i)) / s2]; %#ok<AGROW>
        elseif direction == 2
            s1 = max([rmax2 * sp(i), sp(j), 1]);
            s2 = max([sp(j), rmin2 * sp(i), 1]);
            sd = max(sq(e) * max([sp(i), sp(j)]), 1);
            inequality = [inequality; ...
                (rmax2 * p2(i) - p2(j)) / s1; ...
                (p2(j) - rmin2 * p2(i)) / s2; ...
                -q(e) * (p2(i) - p2(j)) / sd]; %#ok<AGROW>
        end
    end

    if opts.pop_ball_constraints
        for v = 1:model.nsdp
            inequality = [inequality; 1 - z(v)^2]; %#ok<AGROW>
        end
    end
end

% ======================================================================
% Independent numeric POP check and extraction diagnostics
% ======================================================================

function check = local_numeric_pop(x, nlp, model)
    data = nlp.data;
    idx = nlp.index;
    opts = nlp.options;
    matrices = nlp.model_matrices;
    x = x(:);
    if numel(x) ~= model.nsdp
        error('run_matgas_gasmodels_sdp:NumericLength', ...
            'Lifted vector length %d does not equal %d.', ...
            numel(x), model.nsdp);
    end
    z = x ./ model.scale;
    xb = x(1:model.nbase);
    p2 = xb(idx.p2);
    q = xb(idx.q);
    sp = model.scale(idx.p2);
    sq = model.scale(idx.q);

    priority = data.gasmodels_ls.dispatchable_delivery.priority;
    f = priority.' * xb(idx.shed);
    h = (matrices.Aeq * xb - matrices.beq) ./ model.balance_scale;
    for k = 1:model.npipe
        e = data.pipe_idx(k);
        i = data.from(e);
        j = data.to(e);
        a = x(model.i_qabs(k));
        sa = model.scale(model.i_qabs(k));
        Kbar = data.res_over_pa2(e);
        sw = max([sp(i), sp(j), abs(Kbar) * sq(e) * sa, 1]);
        sab = max([sa^2, sq(e)^2, 1]);
        h = [h; ...
            (p2(i) - p2(j) - Kbar * q(e) * a) / sw; ...
            (a^2 - q(e)^2) / sab]; %#ok<AGROW>
    end
    for k = 1:numel(data.short_idx)
        e = data.short_idx(k);
        i = data.from(e);
        j = data.to(e);
        h = [h; (p2(i) - p2(j)) / max([sp(i), sp(j), 1])]; %#ok<AGROW>
    end

    g = [z - model.lb_hat; model.ub_hat - z];
    if opts.sdp_strengthen
        for k = 1:model.npipe
            e = data.pipe_idx(k);
            i = data.from(e);
            j = data.to(e);
            a = x(model.i_qabs(k));
            sa = model.scale(model.i_qabs(k));
            sf = max([sa, sq(e), 1]);
            sd = max(sq(e) * max([sp(i), sp(j)]), 1);
            g = [g; (a - q(e)) / sf; (a + q(e)) / sf; ...
                q(e) * (p2(i) - p2(j)) / sd]; %#ok<AGROW>
        end
    end

    gm = data.gasmodels_ls;
    for k = 1:numel(gm.compressor_arc_index)
        e = gm.compressor_arc_index(k);
        i = data.from(e);
        j = data.to(e);
        rmin2 = data.crmin(e)^2;
        rmax2 = data.crmax(e)^2;
        direction = gm.compressor_directionality(e);
        if direction == 0
            span2 = rmax2 / rmin2;
            g = [g; ...
                (span2 * p2(i) - p2(j)) / max([span2 * sp(i), sp(j), 1]); ...
                (span2 * p2(j) - p2(i)) / max([span2 * sp(j), sp(i), 1]); ...
                -q(e) * (p2(i) - p2(j)) ...
                    / max(sq(e) * max([sp(i), sp(j)]), 1)]; %#ok<AGROW>
        elseif direction == 1
            g = [g; ...
                (rmax2 * p2(i) - p2(j)) / max([rmax2 * sp(i), sp(j), 1]); ...
                (p2(j) - rmin2 * p2(i)) / max([sp(j), rmin2 * sp(i), 1])]; %#ok<AGROW>
        elseif direction == 2
            g = [g; ...
                (rmax2 * p2(i) - p2(j)) / max([rmax2 * sp(i), sp(j), 1]); ...
                (p2(j) - rmin2 * p2(i)) / max([sp(j), rmin2 * sp(i), 1]); ...
                -q(e) * (p2(i) - p2(j)) ...
                    / max(sq(e) * max([sp(i), sp(j)]), 1)]; %#ok<AGROW>
        end
    end
    if opts.pop_ball_constraints
        g = [g; 1 - z.^2];
    end

    check = struct();
    check.objective = f;
    check.minimum_inequality = min([g; Inf]);
    check.max_inequality_violation = max([-g; 0]);
    check.max_equality_violation = max([abs(h); 0]);
    check.max_violation = max(check.max_inequality_violation, ...
        check.max_equality_violation);
    check.inequality_count = numel(g);
    check.equality_count = numel(h);
end

% ======================================================================
% CSTSS recovery and optional coefficient diagnostics
% ======================================================================

function [moment_rpt, ok] = local_moment_representations(aux_info, if_mex)
    moment_rpt = {};
    ok = false;
    needed = {'cliques', 'ts_info'};
    for k = 1:numel(needed)
        if ~isfield(aux_info, needed{k})
            return;
        end
    end
    ts_info = aux_info.ts_info;
    count = 0;
    for i = 1:numel(aux_info.cliques)
        count = count + numel(ts_info{i});
    end
    moment_rpt = cell(count, 1);
    cursor = 0;
    for i = 1:numel(aux_info.cliques)
        for j = 1:numel(ts_info{i})
            cursor = cursor + 1;
            if if_mex
                if ~isfield(aux_info, 'mon_rpt')
                    moment_rpt = {};
                    return;
                end
                rpt = aux_info.mon_rpt{i}(ts_info{i}{j}, :);
                moment_rpt{cursor} = [zeros(size(rpt)), rpt];
            else
                if exist('find_rpt', 'file') == 0
                    moment_rpt = {};
                    return;
                end
                count_here = numel(ts_info{i}{j});
                moment_rpt{cursor} = find_rpt( ...
                    cursor * ones(1, count_here), ones(1, count_here), ...
                    1:count_here, aux_info);
            end
        end
    end
    ok = ~isempty(moment_rpt);
end

function stats = local_pop_coefficient_ranges( ...
        z, objective, inequality, equality, kappa, verbose)
    stats = struct();
    stats.degree_cap = 2 * kappa;
    if exist('supp_rpt', 'file') == 0
        warning('run_matgas_gasmodels_sdp:MissingSuppRpt', ...
            'supp_rpt was not found; coefficient reporting was skipped.');
        stats.status = 'skipped_no_supp_rpt';
        return;
    end
    stats.status = 'computed';
    stats.objective = local_polynomial_coeff_block( ...
        objective, z, stats.degree_cap);
    stats.inequality = local_polynomial_coeff_block( ...
        inequality, z, stats.degree_cap);
    stats.equality = local_polynomial_coeff_block( ...
        equality, z, stats.degree_cap);
    if verbose
        fprintf('POP coefficient |c| ranges (nonzero):\n');
        fprintf('  objective : [%.3e, %.3e]\n', ...
            stats.objective.min_abs_nonzero, stats.objective.max_abs);
        fprintf('  inequality: [%.3e, %.3e]\n', ...
            stats.inequality.min_abs_nonzero, stats.inequality.max_abs);
        fprintf('  equality  : [%.3e, %.3e]\n', ...
            stats.equality.min_abs_nonzero, stats.equality.max_abs);
    end
end

function block = local_polynomial_coeff_block(polynomials, z, degree_cap)
    values = [];
    for k = 1:numel(polynomials)
        [~, c] = supp_rpt(polynomials(k), z, degree_cap);
        values = [values; full(double(c(:)))]; %#ok<AGROW>
    end
    nonzero = abs(values(values ~= 0));
    block = struct();
    block.row_count = numel(polynomials);
    if isempty(nonzero)
        block.min_abs_nonzero = 0;
        block.max_abs = 0;
        block.dynamic_range = 0;
    else
        block.min_abs_nonzero = min(nonzero);
        block.max_abs = max(nonzero);
        block.dynamic_range = block.max_abs / block.min_abs_nonzero;
    end
end

function [known, ok, text] = local_mosek_solution_status(res)
    known = false;
    ok = false;
    text = 'unavailable';
    if ~isstruct(res) || ~isfield(res, 'sol') || ~isstruct(res.sol)
        return;
    end
    solution_types = {'itr', 'int', 'bas'};
    for k = 1:numel(solution_types)
        name = solution_types{k};
        if ~isfield(res.sol, name) || ~isstruct(res.sol.(name)) || ...
                ~isfield(res.sol.(name), 'solsta')
            continue;
        end
        value = res.sol.(name).solsta;
        if ischar(value) || (isstring(value) && isscalar(value))
            text = upper(char(value));
        elseif isnumeric(value) && isscalar(value)
            text = sprintf('NUMERIC_STATUS_%g', value);
            known = false;
            return;
        else
            text = 'UNRECOGNIZED_STATUS';
            known = false;
            return;
        end
        known = true;
        ok = contains(text, 'OPTIMAL') && ~contains(text, 'UNKNOWN');
        return;
    end
end

function local_validate_input(nlp)
    required = {'data', 'index', 'options', 'x', ...
        'objective_weighted_shed', 'model_matrices'};
    for k = 1:numel(required)
        if ~isfield(nlp, required{k})
            error('run_matgas_gasmodels_sdp:Input', ...
                'Input is missing field %s.', required{k});
        end
    end
end
