function out = run_matgas_gasmodels_ls(matgas_file, opts)
%RUN_MATGAS_GASMODELS_LS Solve MatGas load shedding like GasModels WP.
%
%   out = run_matgas_gasmodels_ls(matgas_file)
%   out = run_matgas_gasmodels_ls(matgas_file, opts)
%
% This routine is separate from run_matgas_mgsp.m.  It implements the
% boundary semantics used by GasModels build_ls:
%   * nondispatchable boundaries are fixed at nominal values;
%   * every dispatchable receipt/delivery/transfer remains separate;
%   * dispatchable receipts and transfers use their original min/max;
%   * a dispatchable delivery is served in [minimum, maximum]; and
%   * only prioritized dispatchable delivery enters the objective.
%
% The objective is written as minimum weighted shedding, which is exactly
% equivalent to GasModels' maximum weighted served load.  Squared pressure
% is used directly.  The Weymouth equation uses q*abs(q), as GasModels does,
% and the auxiliary pressure of a continuous type-0 compressor is projected
% out analytically into three equivalent inequalities.
%
% Requirements: Optimization Toolbox (fmincon; linprog is used for starts).
%
% Useful opts fields
%   verbose                    default true
%   num_starts                 default 8
%   random_seed                default 7
%   start_flow_fraction        default 0.01 of each arc flow scale
%   feasibility_tolerance     default 1e-6 (scaled constraints)
%   algorithm                 default 'sqp'
%   fmincon_display            default 'off'
%   max_iterations             default 3000
%   max_function_evaluations  default 500000
%   fmincon_options            optional complete options object or struct
%   return_least_infeasible    default false
%   run_sdp                    default false; driver enables it
%   kappa                     SDP moment relaxation order, default 2
%   cs_mode                   CSTSS correlative sparsity mode, default 'MF'
%   pop_ball_constraints      add per-variable normalized balls, default true
%   sdp_strengthen            add redundant pipe inequalities, default true
%   report_pop_coeff_ranges   print POP coefficient ranges, default false
%   save_file                  optional .mat output path

    if nargin < 1 || isempty(matgas_file)
        error('run_matgas_gasmodels_ls:MissingFile', ...
            'A MatGas .m file is required.');
    end
    if nargin < 2 || isempty(opts)
        opts = struct();
    end
    opts = local_defaults(opts);
    local_validate_options(opts);

    if exist('fmincon', 'file') ~= 2
        error('run_matgas_gasmodels_ls:MissingFmincon', ...
            'Optimization Toolbox function fmincon is required.');
    end

    data = parse_matgas_gasmodels_ls(matgas_file, opts);
    local_validate_supported_physics(data);

    idx = local_build_index(data);
    [Aeq, beq] = local_build_mass_balance(data, idx);
    [lb, ub] = local_build_bounds(data, idx);
    scale = local_variable_scale(lb, ub);

    Dscale = spdiags(scale, 0, idx.nx, idx.nx);
    Aeq_hat = Aeq * Dscale;
    balance_scale = max([full(sum(abs(Aeq_hat), 2)), abs(beq), ...
        ones(data.N, 1)], [], 2);
    Aeq_hat = spdiags(1 ./ balance_scale, 0, data.N, data.N) * Aeq_hat;
    % fmincon's SQP implementation does not accept sparse linear-constraint
    % matrices and otherwise emits a conversion warning on every run.
    % This matrix is small (N-by-nx), so convert it once here explicitly.
    Aeq_hat = full(Aeq_hat);
    beq_hat = beq ./ balance_scale;
    lb_hat = lb ./ scale;
    ub_hat = ub ./ scale;

    objective_coeff = zeros(idx.nx, 1);
    objective_coeff(idx.shed) = ...
        data.gasmodels_ls.dispatchable_delivery.priority;
    objective_coeff_hat = objective_coeff .* scale;
    objective_coeff_hat = objective_coeff_hat ...
        / max(norm(objective_coeff_hat, Inf), 1);

    fopts = local_fmincon_options(opts);
    old_rng = rng;
    rng_cleanup = onCleanup(@() rng(old_rng)); %#ok<NASGU>
    rng(opts.random_seed, 'twister');

    blank = struct('x', [], 'y', [], 'objective', Inf, 'exitflag', NaN, ...
        'output', struct(), 'is_feasible', false, ...
        'max_scaled_violation', Inf, 'residuals', struct(), ...
        'start', [], 'error_message', '');
    runs = repmat(blank, opts.num_starts, 1);

    if opts.verbose
        fprintf('\n=== GasModels-compatible nonlinear load shedding ===\n');
        fprintf('Variables                    : %d\n', idx.nx);
        fprintf('Starts                       : %d\n', opts.num_starts);
        fprintf('Objective                    : min weighted dispatchable shed\n');
    end

    for start_number = 1:opts.num_starts
        x0 = local_make_start(start_number, data, idx, Aeq, beq, ...
            lb, ub, scale, objective_coeff, opts);
        y0 = x0 ./ scale;
        runs(start_number).start = x0;

        try
            [y, ~, exitflag, solver_output] = fmincon( ...
                @(yh) local_objective(yh, objective_coeff_hat), ...
                y0, [], [], Aeq_hat, beq_hat, lb_hat, ub_hat, ...
                @(yh) local_nonlcon_hat(yh, scale, data, idx), fopts);
            x = y .* scale;
            residuals = local_residuals(x, y, data, idx, Aeq, beq, ...
                Aeq_hat, beq_hat, lb, ub, scale);

            runs(start_number).x = x;
            runs(start_number).y = y;
            runs(start_number).objective = objective_coeff.' * x;
            runs(start_number).exitflag = exitflag;
            runs(start_number).output = solver_output;
            runs(start_number).residuals = residuals;
            runs(start_number).max_scaled_violation = ...
                residuals.max_scaled_violation;
            runs(start_number).is_feasible = ...
                residuals.max_scaled_violation <= opts.feasibility_tolerance;
        catch ME
            runs(start_number).error_message = ME.message;
            if opts.verbose
                warning('run_matgas_gasmodels_ls:StartFailed', ...
                    'Start %d failed: %s', start_number, ME.message);
            end
        end

        if opts.verbose && ~isempty(runs(start_number).x)
            fprintf(['Start %2d/%d: shed=%12.6f  violation=%9.3e  ', ...
                'exitflag=%d  feasible=%d\n'], start_number, opts.num_starts, ...
                runs(start_number).objective, ...
                runs(start_number).max_scaled_violation, ...
                runs(start_number).exitflag, ...
                runs(start_number).is_feasible);
        end
    end

    completed = find(~cellfun(@isempty, {runs.x}));
    if isempty(completed)
        error('run_matgas_gasmodels_ls:AllStartsFailed', ...
            'Every fmincon start failed before returning a point.');
    end
    feasible = completed([runs(completed).is_feasible]);
    if ~isempty(feasible)
        [~, loc] = min([runs(feasible).objective]);
        best_index = feasible(loc);
    else
        [~, loc] = min([runs(completed).max_scaled_violation]);
        best_index = completed(loc);
        if ~opts.return_least_infeasible
            error('run_matgas_gasmodels_ls:NoFeasibleStart', ...
                ['No start met the scaled feasibility tolerance %.3e; ', ...
                 'the least violation was %.3e.  Increase num_starts or ', ...
                 'set return_least_infeasible=true to inspect that point.'], ...
                opts.feasibility_tolerance, ...
                runs(best_index).max_scaled_violation);
        end
        warning('run_matgas_gasmodels_ls:NoFeasibleStart', ...
            ['No start met the scaled feasibility tolerance %.3e.  ', ...
             'Returning the least-infeasible point because ', ...
             'return_least_infeasible=true.'], opts.feasibility_tolerance);
    end

    out = local_build_output(data, idx, runs, best_index, ...
        objective_coeff, Aeq, beq, lb, ub, scale, opts);

    if opts.verbose
        local_print_summary(out);
    end

    if opts.run_sdp
        try
            out.sdp = run_matgas_gasmodels_sdp(out);
        catch ME
            if opts.sdp_fail_on_error
                rethrow(ME);
            end
            warning('run_matgas_gasmodels_ls:SdpFailed', ...
                'SDP relaxation failed after fmincon succeeded: %s', ME.message);
            out.sdp = struct('status', 'failed', ...
                'error_identifier', ME.identifier, ...
                'error_message', ME.message);
        end
    else
        out.sdp = struct('status', 'not_requested');
    end

    if ~isempty(opts.save_file)
        save(char(opts.save_file), 'out', '-v7.3');
    end
end

% ======================================================================
% Model construction
% ======================================================================

function idx = local_build_index(data)
    gm = data.gasmodels_ls;
    idx = struct();
    next = 0;

    idx.p2 = next + (1:data.N); next = next + data.N;
    idx.q = next + (1:data.M); next = next + data.M;
    idx.receipt = next + (1:numel(gm.dispatchable_receipt.id));
    next = next + numel(gm.dispatchable_receipt.id);
    idx.shed = next + (1:numel(gm.dispatchable_delivery.id));
    next = next + numel(gm.dispatchable_delivery.id);
    idx.transfer = next + (1:numel(gm.dispatchable_transfer.id));
    next = next + numel(gm.dispatchable_transfer.id);

    idx.nx = next;
end

function [Aeq, beq] = local_build_mass_balance(data, idx)
    gm = data.gasmodels_ls;
    Aeq = sparse(data.N, idx.nx);
    for e = 1:data.M
        Aeq(data.from(e), idx.q(e)) = ...
            Aeq(data.from(e), idx.q(e)) + 1;
        Aeq(data.to(e), idx.q(e)) = ...
            Aeq(data.to(e), idx.q(e)) - 1;
    end
    for k = 1:numel(idx.receipt)
        n = gm.dispatchable_receipt.node_index(k);
        Aeq(n, idx.receipt(k)) = Aeq(n, idx.receipt(k)) - 1;
    end
    for k = 1:numel(idx.shed)
        n = gm.dispatchable_delivery.node_index(k);
        Aeq(n, idx.shed(k)) = Aeq(n, idx.shed(k)) - 1;
    end
    for k = 1:numel(idx.transfer)
        n = gm.dispatchable_transfer.node_index(k);
        Aeq(n, idx.transfer(k)) = Aeq(n, idx.transfer(k)) + 1;
    end
    beq = gm.balance_rhs;
end

function [lb, ub] = local_build_bounds(data, idx)
    gm = data.gasmodels_ls;
    lb = -inf(idx.nx, 1);
    ub =  inf(idx.nx, 1);

    lb(idx.p2) = data.pmin_bar .^ 2;
    ub(idx.p2) = data.pmax_bar .^ 2;
    lb(idx.q) = data.qmin;
    ub(idx.q) = data.qmax;

    lb(idx.receipt) = gm.dispatchable_receipt.minimum;
    ub(idx.receipt) = gm.dispatchable_receipt.maximum;
    lb(idx.shed) = 0;
    ub(idx.shed) = gm.dispatchable_delivery.shed_max;
    lb(idx.transfer) = gm.dispatchable_transfer.minimum;
    ub(idx.transfer) = gm.dispatchable_transfer.maximum;

    if any(~isfinite(lb) | ~isfinite(ub) | lb > ub)
        error('run_matgas_gasmodels_ls:VariableBounds', ...
            'All variable bounds must be finite and nonempty.');
    end
end

function scale = local_variable_scale(lb, ub)
    scale = max([abs(lb), abs(ub), ones(size(lb))], [], 2);
end

% ======================================================================
% Objective and nonlinear constraints
% ======================================================================

function [f, gradient] = local_objective(y, coeff)
    f = coeff.' * y;
    if nargout > 1
        gradient = coeff;
    end
end

function [c, ceq] = local_nonlcon_hat(y, variable_scale, data, idx)
    x = y .* variable_scale;
    [c_physical, ceq_physical, c_scale, ceq_scale] = ...
        local_physical_constraints(x, variable_scale, data, idx);
    c = c_physical ./ c_scale;
    ceq = ceq_physical ./ ceq_scale;
end

function [c, ceq, c_scale, ceq_scale] = ...
        local_physical_constraints(x, variable_scale, data, idx)
    p2 = x(idx.p2);
    q = x(idx.q);
    sp = variable_scale(idx.p2);
    sq = variable_scale(idx.q);

    c = zeros(0, 1);
    c_scale = zeros(0, 1);
    ceq = zeros(0, 1);
    ceq_scale = zeros(0, 1);

    % GasModels WPGasModel uses this direct Weymouth equation.  Unlike the
    % old qabs^2=q^2 lifting, its equality gradient never becomes an all-zero
    % row at q=0 because the two pressure derivatives remain +1 and -1.
    for k = 1:numel(data.pipe_idx)
        e = data.pipe_idx(k);
        i = data.from(e);
        j = data.to(e);
        Kbar = data.res_over_pa2(e);
        ceq(end+1, 1) = p2(i) - p2(j) ...
            - Kbar * q(e) * abs(q(e)); %#ok<AGROW>
        ceq_scale(end+1, 1) = max([sp(i), sp(j), ...
            abs(Kbar) * sq(e)^2, 1]); %#ok<AGROW>
    end

    for k = 1:numel(data.short_idx)
        e = data.short_idx(k);
        i = data.from(e);
        j = data.to(e);
        ceq(end+1, 1) = p2(i) - p2(j); %#ok<AGROW>
        ceq_scale(end+1, 1) = max([sp(i), sp(j), 1]); %#ok<AGROW>
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
            % Exact projection of GasModels' six auxiliary-pk constraints
            % for rmin <= 1 <= rmax.  For gaslib-40-E-ls, rmin=1 and
            % ratio_span2 is simply 5^2=25.
            ratio_span2 = rmax2 / rmin2;
            c(end+1, 1) = p2(j) - ratio_span2 * p2(i); %#ok<AGROW>
            c_scale(end+1, 1) = max([sp(j), ratio_span2 * sp(i), 1]); %#ok<AGROW>
            c(end+1, 1) = p2(i) - ratio_span2 * p2(j); %#ok<AGROW>
            c_scale(end+1, 1) = max([sp(i), ratio_span2 * sp(j), 1]); %#ok<AGROW>
            c(end+1, 1) = q(e) * (p2(i) - p2(j)); %#ok<AGROW>
            c_scale(end+1, 1) = max(sq(e) * max([sp(i), sp(j)]), 1); %#ok<AGROW>
        elseif direction == 1
            c(end+1, 1) = p2(j) - rmax2 * p2(i); %#ok<AGROW>
            c_scale(end+1, 1) = max([sp(j), rmax2 * sp(i), 1]); %#ok<AGROW>
            c(end+1, 1) = rmin2 * p2(i) - p2(j); %#ok<AGROW>
            c_scale(end+1, 1) = max([rmin2 * sp(i), sp(j), 1]); %#ok<AGROW>
        elseif direction == 2
            c(end+1, 1) = p2(j) - rmax2 * p2(i); %#ok<AGROW>
            c_scale(end+1, 1) = max([sp(j), rmax2 * sp(i), 1]); %#ok<AGROW>
            c(end+1, 1) = rmin2 * p2(i) - p2(j); %#ok<AGROW>
            c_scale(end+1, 1) = max([rmin2 * sp(i), sp(j), 1]); %#ok<AGROW>
            c(end+1, 1) = q(e) * (p2(i) - p2(j)); %#ok<AGROW>
            c_scale(end+1, 1) = max(sq(e) * max([sp(i), sp(j)]), 1); %#ok<AGROW>
        else
            error('run_matgas_gasmodels_ls:Directionality', ...
                'Unexpected compressor directionality %.9g.', direction);
        end
    end
end

% ======================================================================
% Starts and solver options
% ======================================================================

function x0 = local_make_start(start_number, data, idx, Aeq, beq, ...
        lb, ub, scale, objective_coeff, opts)
    x0 = 0.5 * (lb + ub);
    span = ub(idx.p2) - lb(idx.p2);
    if start_number == 1
        alpha = 0.5 * ones(data.N, 1);
    elseif start_number == 2 || start_number == 3
        alpha = zeros(data.N, 1);
    else
        alpha = rand(data.N, 1);
    end
    x0(idx.p2) = lb(idx.p2) + alpha .* span;

    linear_index = [idx.q(:); idx.receipt(:); idx.shed(:); idx.transfer(:)];
    [linear_start, ok] = local_linear_start(start_number, ...
        Aeq(:, linear_index), beq, lb(linear_index), ub(linear_index), ...
        scale(linear_index), objective_coeff(linear_index), opts);
    if ok
        x0(linear_index) = linear_start;
    end

    % A linear-program basic solution commonly sets many cycle flows to
    % exactly zero.  At q=0 the Weymouth flow derivative is zero, so the
    % pipe-pressure incidence rows can become rank deficient on cycles.
    % A small bounded nudge gives fmincon a regular initial Jacobian.  The
    % start need not remain exactly mass-balanced; fmincon restores balance.
    x0(idx.q) = local_nudge_zero_flows(x0(idx.q), ...
        lb(idx.q), ub(idx.q), scale(idx.q), start_number, opts);
    x0 = min(max(x0, lb), ub);
end

function q = local_nudge_zero_flows(q, lb, ub, scale, start_number, opts)
    floor_value = opts.start_flow_fraction .* scale;
    for e = 1:numel(q)
        if abs(q(e)) >= floor_value(e) || floor_value(e) == 0
            continue;
        end
        if lb(e) >= 0
            candidate = floor_value(e);
        elseif ub(e) <= 0
            candidate = -floor_value(e);
        elseif mod(e + start_number, 2) == 0
            candidate = floor_value(e);
        else
            candidate = -floor_value(e);
        end
        q(e) = min(max(candidate, lb(e)), ub(e));
    end
end

function [z, ok] = local_linear_start(start_number, Aeq, beq, ...
        lb, ub, scale, objective_coeff, opts)
    z = 0.5 * (lb + ub);
    ok = false;
    if exist('linprog', 'file') ~= 2
        if start_number == 1 && opts.verbose
            warning('run_matgas_gasmodels_ls:MissingLinprog', ...
                ['linprog was not found; fmincon starts will not be ', ...
                 'mass-balance feasible.']);
        end
        return;
    end

    row_scale = max([full(sum(abs(Aeq), 2)), abs(beq), ...
        ones(size(beq))], [], 2);
    Aeq_scaled = spdiags(1 ./ row_scale, 0, numel(beq), numel(beq)) * Aeq;
    beq_scaled = beq ./ row_scale;

    if start_number == 1
        f = objective_coeff;
    else
        f = 1e-3 * randn(size(scale)) ./ scale;
        if mod(start_number, 2) == 0
            f = f + objective_coeff;
        end
    end
    try
        lp_options = optimoptions('linprog', 'Display', 'none');
        [candidate, ~, exitflag] = linprog(f, [], [], Aeq_scaled, ...
            beq_scaled, lb, ub, lp_options);
        if exitflag > 0 && ~isempty(candidate)
            z = candidate;
            ok = true;
        end
    catch ME
        if start_number == 1 && opts.verbose
            warning('run_matgas_gasmodels_ls:LinprogFailed', ...
                'Could not construct a linear feasible start: %s', ME.message);
        end
    end
end

function fopts = local_fmincon_options(opts)
    if ~isempty(opts.fmincon_options) && ~isstruct(opts.fmincon_options)
        fopts = opts.fmincon_options;
        return;
    end

    fopts = optimoptions('fmincon', ...
        'Algorithm', char(opts.algorithm), ...
        'Display', char(opts.fmincon_display), ...
        'MaxIterations', opts.max_iterations, ...
        'MaxFunctionEvaluations', opts.max_function_evaluations, ...
        'ConstraintTolerance', opts.constraint_tolerance, ...
        'OptimalityTolerance', opts.optimality_tolerance, ...
        'StepTolerance', opts.step_tolerance, ...
        'SpecifyObjectiveGradient', true, ...
        'ScaleProblem', false);

    if isstruct(opts.fmincon_options)
        names = fieldnames(opts.fmincon_options);
        for k = 1:numel(names)
            try
                fopts.(names{k}) = opts.fmincon_options.(names{k});
            catch ME
                error('run_matgas_gasmodels_ls:FminconOption', ...
                    'Invalid fmincon option %s: %s', names{k}, ME.message);
            end
        end
    end
end

% ======================================================================
% Validation, residuals, and output
% ======================================================================

function local_validate_supported_physics(data)
    gm = data.gasmodels_ls;
    for k = 1:numel(gm.compressor_arc_index)
        e = gm.compressor_arc_index(k);
        direction = gm.compressor_directionality(e);
        rmin = data.crmin(e);
        rmax = data.crmax(e);
        if rmin <= 0
            error('run_matgas_gasmodels_ls:CompressorRatio', ...
                'Compressor %s must have a strictly positive minimum ratio.', ...
                data.arc_ids{e});
        end
        if direction == 0 && ~(rmin <= 1 && rmax >= 1)
            error('run_matgas_gasmodels_ls:DiscreteCompressor', ...
                ['Type-0 compressor %s requires GasModels binary ', ...
                 'direction logic because its ratio interval does not ', ...
                 'contain 1.  This continuous routine will not approximate it.'], ...
                data.arc_ids{e});
        end
        if direction == 2 && rmin ~= 1
            error('run_matgas_gasmodels_ls:DiscreteCompressor', ...
                ['Type-2 compressor %s has minimum ratio different from 1 ', ...
                 'and requires a discrete GasModels formulation.'], ...
                data.arc_ids{e});
        end
    end
end

function residuals = local_residuals(x, y, data, idx, Aeq, beq, ...
        Aeq_hat, beq_hat, lb, ub, variable_scale)
    [c_raw, ceq_raw, c_scale, ceq_scale] = ...
        local_physical_constraints(x, variable_scale, data, idx);
    c_hat = c_raw ./ c_scale;
    ceq_hat = ceq_raw ./ ceq_scale;
    balance_raw = Aeq * x - beq;
    balance_hat = Aeq_hat * y - beq_hat;
    lower_violation = max(lb - x, 0) ./ variable_scale;
    upper_violation = max(x - ub, 0) ./ variable_scale;

    residuals = struct();
    residuals.balance_kgps = balance_raw;
    residuals.nonlinear_inequality_raw = c_raw;
    residuals.nonlinear_equality_raw = ceq_raw;
    residuals.max_balance_kgps = max([abs(balance_raw); 0]);
    residuals.max_nonlinear_inequality_raw = max([c_raw; 0]);
    residuals.max_nonlinear_equality_raw = max([abs(ceq_raw); 0]);
    residuals.max_scaled_balance = max([abs(balance_hat); 0]);
    residuals.max_scaled_nonlinear_inequality = max([c_hat; 0]);
    residuals.max_scaled_nonlinear_equality = max([abs(ceq_hat); 0]);
    residuals.max_scaled_bound_violation = max([lower_violation; ...
        upper_violation; 0]);
    residuals.max_scaled_violation = max([ ...
        residuals.max_scaled_balance, ...
        residuals.max_scaled_nonlinear_inequality, ...
        residuals.max_scaled_nonlinear_equality, ...
        residuals.max_scaled_bound_violation]);
end

function out = local_build_output(data, idx, runs, best_index, ...
        objective_coeff, Aeq, beq, lb, ub, scale, opts)
    best = runs(best_index);
    x = best.x;
    gm = data.gasmodels_ls;

    p2 = x(idx.p2);
    pressure = sqrt(max(p2, 0));
    flow = x(idx.q);
    receipt = x(idx.receipt);
    shed = x(idx.shed);
    served = gm.dispatchable_delivery.maximum - shed;
    transfer = x(idx.transfer);

    out = struct();
    out.formulation = gm.formulation;
    out.data = data;
    out.options = opts;
    out.index = idx;
    out.best_start = best_index;
    out.runs = runs;
    out.x = x;
    out.exitflag = best.exitflag;
    out.solver_output = best.output;
    out.is_feasible = best.is_feasible;
    out.residuals = best.residuals;
    out.objective_weighted_shed = objective_coeff.' * x;

    out.junction = table((1:data.N).', string(data.node_ids(:)), ...
        p2, pressure, data.pmin_bar, data.pmax_bar, ...
        'VariableNames', {'index', 'id', 'pressure_bar2', 'pressure_bar', ...
        'minimum_bar', 'maximum_bar'});
    out.arc = table((1:data.M).', string(data.arc_ids(:)), ...
        string(data.arc_types(:)), string(data.matgas.arc_subtypes(:)), ...
        data.from, data.to, flow, data.qmin, data.qmax, ...
        'VariableNames', {'index', 'id', 'type', 'subtype', 'from', 'to', ...
        'flow_kgps', 'minimum_kgps', 'maximum_kgps'});

    rnode = gm.dispatchable_receipt.node_index;
    out.dispatchable_receipt = table(string(gm.dispatchable_receipt.id), ...
        string(data.node_ids(rnode)), receipt, ...
        gm.dispatchable_receipt.minimum, gm.dispatchable_receipt.maximum, ...
        'VariableNames', {'id', 'junction_id', 'injection_kgps', ...
        'minimum_kgps', 'maximum_kgps'});

    dnode = gm.dispatchable_delivery.node_index;
    out.dispatchable_delivery = table(string(gm.dispatchable_delivery.id), ...
        string(data.node_ids(dnode)), served, shed, ...
        gm.dispatchable_delivery.minimum, gm.dispatchable_delivery.maximum, ...
        gm.dispatchable_delivery.priority, ...
        'VariableNames', {'id', 'junction_id', 'served_kgps', 'shed_kgps', ...
        'minimum_kgps', 'maximum_kgps', 'priority'});

    tnode = gm.dispatchable_transfer.node_index;
    out.dispatchable_transfer = table(string(gm.dispatchable_transfer.id), ...
        string(data.node_ids(tnode)), transfer, ...
        gm.dispatchable_transfer.minimum, gm.dispatchable_transfer.maximum, ...
        'VariableNames', {'id', 'junction_id', 'withdrawal_kgps', ...
        'minimum_kgps', 'maximum_kgps'});

    summary = struct();
    summary.fixed_receipt_kgps = gm.fixed_supply_total;
    summary.fixed_delivery_kgps = gm.fixed_demand_total;
    summary.fixed_transfer_withdrawal_kgps = gm.fixed_transfer_total;
    summary.dispatchable_receipt_kgps = sum(receipt);
    summary.dispatchable_receipt_capacity_kgps = ...
        gm.dispatchable_supply_max;
    summary.dispatchable_delivery_target_kgps = ...
        gm.dispatchable_demand_max;
    summary.dispatchable_delivery_served_kgps = sum(served);
    summary.dispatchable_delivery_shed_kgps = sum(shed);
    summary.dispatchable_transfer_withdrawal_kgps = sum(transfer);
    summary.weighted_dispatchable_target = ...
        gm.dispatchable_delivery.priority.' ...
        * gm.dispatchable_delivery.maximum;
    summary.weighted_dispatchable_served = ...
        gm.dispatchable_delivery.priority.' * served;
    summary.weighted_dispatchable_shed = ...
        gm.dispatchable_delivery.priority.' * shed;
    summary.total_receipt_kgps = gm.fixed_supply_total + sum(receipt);
    summary.total_delivery_kgps = gm.fixed_demand_total + sum(served);
    summary.total_transfer_withdrawal_kgps = ...
        gm.fixed_transfer_total + sum(transfer);
    if gm.dispatchable_demand_max > 0
        summary.dispatchable_served_percent = ...
            100 * sum(served) / gm.dispatchable_demand_max;
    else
        summary.dispatchable_served_percent = NaN;
    end
    out.summary = summary;

    [~, base, extension] = fileparts(char(data.net_file));
    if strcmpi([base, extension], 'gaslib-40-E-ls.m')
        expected = struct();
        expected.source = 'GasModels WPGasModel regression (locally solved)';
        expected.dispatchable_served_kgps = 420.91;
        expected.dispatchable_shed_kgps = 1324.09;
        expected.dispatchable_receipt_kgps = 426.91;
        expected.served_difference_kgps = ...
            summary.dispatchable_delivery_served_kgps ...
            - expected.dispatchable_served_kgps;
        expected.shed_difference_kgps = ...
            summary.dispatchable_delivery_shed_kgps ...
            - expected.dispatchable_shed_kgps;
        out.gasmodels_benchmark = expected;
    else
        out.gasmodels_benchmark = struct();
    end

    out.model_matrices = struct('Aeq', Aeq, 'beq', beq, ...
        'lb', lb, 'ub', ub, 'variable_scale', scale);
end

function local_print_summary(out)
    s = out.summary;
    fprintf('\n=== Best solution ===\n');
    fprintf('Best start                   : %d\n', out.best_start);
    fprintf('Feasible at requested tol    : %d\n', out.is_feasible);
    fprintf('Max scaled violation         : %.3e\n', ...
        out.residuals.max_scaled_violation);
    fprintf('Fixed receipt / delivery     : %.9g / %.9g kg/s\n', ...
        s.fixed_receipt_kgps, s.fixed_delivery_kgps);
    fprintf('Dispatch receipt used        : %.9g kg/s\n', ...
        s.dispatchable_receipt_kgps);
    fprintf('Dispatch delivery target     : %.9g kg/s\n', ...
        s.dispatchable_delivery_target_kgps);
    fprintf('Dispatch delivery served     : %.9g kg/s\n', ...
        s.dispatchable_delivery_served_kgps);
    fprintf('Dispatch delivery shed       : %.9g kg/s\n', ...
        s.dispatchable_delivery_shed_kgps);
    fprintf('Dispatch served              : %.4f %%\n', ...
        s.dispatchable_served_percent);
    fprintf('Weighted shed objective      : %.9g\n', ...
        s.weighted_dispatchable_shed);
    if ~isempty(fieldnames(out.gasmodels_benchmark)) && out.is_feasible
        b = out.gasmodels_benchmark;
        fprintf('\nGasModels WP served reference: %.9g kg/s\n', ...
            b.dispatchable_served_kgps);
        fprintf('MATLAB minus GasModels served: %+.9g kg/s\n', ...
            b.served_difference_kgps);
        fprintf(['Reference is a locally solved regression value, ', ...
            'not a certified global optimum.\n']);
    elseif ~isempty(fieldnames(out.gasmodels_benchmark))
        fprintf('\nBenchmark comparison skipped: returned point is infeasible.\n');
    end
end

function opts = local_defaults(opts)
    opts = local_set_default(opts, 'verbose', true);
    opts = local_set_default(opts, 'num_starts', 8);
    opts = local_set_default(opts, 'random_seed', 7);
    opts = local_set_default(opts, 'start_flow_fraction', 0.01);
    opts = local_set_default(opts, 'feasibility_tolerance', 1e-6);
    opts = local_set_default(opts, 'bound_tolerance', 1e-9);
    opts = local_set_default(opts, 'algorithm', 'sqp');
    opts = local_set_default(opts, 'fmincon_display', 'off');
    opts = local_set_default(opts, 'max_iterations', 3000);
    opts = local_set_default(opts, 'max_function_evaluations', 500000);
    opts = local_set_default(opts, 'constraint_tolerance', 1e-8);
    opts = local_set_default(opts, 'optimality_tolerance', 1e-8);
    opts = local_set_default(opts, 'step_tolerance', 1e-12);
    opts = local_set_default(opts, 'fmincon_options', []);
    opts = local_set_default(opts, 'return_least_infeasible', false);
    opts = local_set_default(opts, 'run_sdp', false);
    opts = local_set_default(opts, 'kappa', 2);
    opts = local_set_default(opts, 'cs_mode', 'MF');
    opts = local_set_default(opts, 'pop_feas_tol', 1e-7);
    opts = local_set_default(opts, 'pop_ball_constraints', true);
    opts = local_set_default(opts, 'sdp_strengthen', true);
    opts = local_set_default(opts, 'report_pop_coeff_ranges', false);
    opts = local_set_default(opts, 'sdp_fail_on_error', false);
    opts = local_set_default(opts, 'save_file', '');
end

function local_validate_options(opts)
    if ~(isscalar(opts.num_starts) && isfinite(opts.num_starts) && ...
            opts.num_starts >= 1 && opts.num_starts == round(opts.num_starts))
        error('run_matgas_gasmodels_ls:Option', ...
            'num_starts must be a positive integer.');
    end
    positive = {'feasibility_tolerance', 'bound_tolerance', ...
        'max_iterations', 'max_function_evaluations', ...
        'constraint_tolerance', 'optimality_tolerance', 'step_tolerance', ...
        'pop_feas_tol'};
    for k = 1:numel(positive)
        value = opts.(positive{k});
        if ~(isscalar(value) && isfinite(value) && value > 0)
            error('run_matgas_gasmodels_ls:Option', ...
                '%s must be a positive finite scalar.', positive{k});
        end
    end
    if ~(isscalar(opts.start_flow_fraction) && ...
            isfinite(opts.start_flow_fraction) && ...
            opts.start_flow_fraction >= 0 && opts.start_flow_fraction <= 0.25)
        error('run_matgas_gasmodels_ls:Option', ...
            'start_flow_fraction must lie between 0 and 0.25.');
    end
    if ~(isscalar(opts.kappa) && isfinite(opts.kappa) && ...
            opts.kappa >= 1 && opts.kappa == round(opts.kappa))
        error('run_matgas_gasmodels_ls:Option', ...
            'kappa must be a positive integer.');
    end
    if ~(isscalar(opts.return_least_infeasible) && ...
            (islogical(opts.return_least_infeasible) || ...
             (isnumeric(opts.return_least_infeasible) && ...
              ismember(opts.return_least_infeasible, [0, 1]))))
        error('run_matgas_gasmodels_ls:Option', ...
            'return_least_infeasible must be true or false.');
    end
    logical_options = {'run_sdp', 'pop_ball_constraints', ...
        'sdp_strengthen', 'report_pop_coeff_ranges', 'sdp_fail_on_error'};
    for k = 1:numel(logical_options)
        value = opts.(logical_options{k});
        if ~(isscalar(value) && (islogical(value) || ...
                (isnumeric(value) && ismember(value, [0, 1]))))
            error('run_matgas_gasmodels_ls:Option', ...
                '%s must be true or false.', logical_options{k});
        end
    end
end

function s = local_set_default(s, name, value)
    if ~isfield(s, name) || isempty(s.(name))
        s.(name) = value;
    end
end
