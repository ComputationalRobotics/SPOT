function data = parse_matgas_instance(matgas_file, opts)
%PARSE_MATGAS_INSTANCE Parse a GasModels MatGas .m case for run_gaslib_mgsp.
%
%   data = parse_matgas_instance(matgas_file)
%   data = parse_matgas_instance(matgas_file, opts)
%
% This parser targets the current MatGas v0.6+ table layout.  It is a
% literal-data parser: it never runs, evals, or fevals the input
% .m file.  That matters because some public MatGas cases use function names
% containing hyphens and because a case file should be treated as data.
%
% The returned struct intentionally matches the output schema of the local
% parse_gaslib_instance() in run_gaslib_mgsp.m.  Thus the following existing
% code can be reused after replacing only the parser call:
%
%   data = parse_matgas_instance(matgas_file, opts);
%   data = build_variable_indexing(data, opts);
%   [Aeq, beq] = build_mass_balance(data);
%   [lb, ub, x0] = build_bounds_and_x0(data);
%
% Supported network components
%   junction, pipe, compressor, short_pipe, valve, regulator,
%   receipt, delivery, and fixed-at-nominal transfer rows.
%
% Component mappings used by the current simplified physical model
%   valve     -> shortPipe (open valve, zero pressure loss)
%   regulator -> compressorStation with pressure ratio <= 1
%
% A regulator mapping is exact only for forward flow under the simplified
% ratio model.  Active resistor, loss_resistor, storage, ne_pipe, and
% ne_compressor rows are rejected because run_gaslib_mgsp has no matching
% constraints.
%
% Important load-shedding semantics
%   * A nondispatchable receipt/delivery uses its nominal amount.
%   * A dispatchable receipt/delivery uses its maximum amount as the capacity
%     or demand target.  This is required for case-6-ls.m, whose dispatchable
%     deliveries have zero nominal withdrawal but positive maxima.
%   * The current run_gaslib_mgsp model aggregates boundaries by junction and
%     then makes every positive demand sheddable and every supply variable
%     range from zero to its cap.  That is not identical to GasModels when a
%     junction mixes fixed and dispatchable boundaries or delivery priorities.
%     The unaggregated information is retained in data.matgas.
%
% Units
%   SI MatGas data are converted to the physical units used by the solver:
%     pressure -> absolute bar, flow -> kg/s, length -> m, diameter -> m.
%   USC data (psi, MMSCFD, miles, inches) and SI/USC per-unit cases are also
%   converted.  For per-unit data, pressure, length, and flow are rescaled by
%   base_pressure, base_length, and base_flow.  MatGas pipe diameters remain
%   dimensional, consistent with GasModels' conversion convention.
%
% Relevant optional fields in opts
%   opts.passive_bidirectional       default true
%   opts.enforce_compressor_forward  default true
%   opts.weight_default              default 1
%   opts.weights                     per-node overrides; fields may be
%                                    node_<id> or MATLAB-valid node ids
%   opts.verbose                     default true
%   opts.matgas_demand_target        'auto' (default), 'nominal', or 'maximum'
%   opts.matgas_supply_target        'auto' (default), 'nominal', or 'maximum'
%   opts.matgas_use_priorities       default true; reads delivery_data priority
%   opts.matgas_fix_slack_pressure   default true; fixes type-1 junctions at
%                                    p_nominal, matching GasModels solve_ls
%   opts.matgas_z_override           empty (default) or scalar override
%   opts.matgas_zero_loss_flow_bound empty (default) or positive kg/s bound
%   opts.matgas_max_pipe_flow_bound  inf (default) or positive kg/s cap
%   opts.matgas_cap_by_network_flow  default false; optional heuristic cap on
%                                    passive/compressor arcs using the zero-loss
%                                    bound (can change the feasible set in cycles)
%   opts.matgas_map_valves           default true
%   opts.matgas_map_regulators       default true
%   opts.matgas_transfer_mode        'error' (default), 'nominal', or 'ignore'
%                                    for active dispatchable transfers
%   opts.matgas_error_on_unsupported default true
%   opts.matgas_error_on_unknown_extensions default true; reject nonempty
%                                    unimplemented mgc.*_data extension tables
%
% Extra diagnostics and original boundary data are stored in data.matgas.

    if nargin < 1 || isempty(matgas_file)
        error('parse_matgas_instance:MissingFile', 'A MatGas .m file is required.');
    end
    if nargin < 2 || isempty(opts)
        opts = struct();
    end
    opts = matgas_default_opts(opts);

    if ~(ischar(matgas_file) || (isstring(matgas_file) && isscalar(matgas_file)))
        error('parse_matgas_instance:BadFile', 'matgas_file must be a character vector or scalar string.');
    end
    matgas_file = char(matgas_file);
    if exist(matgas_file, 'file') ~= 2
        error('parse_matgas_instance:MissingFile', 'MatGas file not found: %s', matgas_file);
    end

    raw_text = fileread(matgas_file);
    raw_text = matgas_normalize_text(raw_text);
    clean_text = matgas_strip_comments(raw_text);

    % ------------------------------------------------------------------
    % Global parameters
    % ------------------------------------------------------------------
    units = lower(matgas_scalar_string(clean_text, 'units', true, ''));
    if ~ismember(units, {'si', 'usc'})
        error('parse_matgas_instance:Units', ...
            'mgc.units must be ''si'' or ''usc''; found ''%s''.', units);
    end

    pu_a = matgas_scalar_number(clean_text, 'is_per_unit', false, NaN);
    pu_b = matgas_scalar_number(clean_text, 'per_unit', false, NaN);
    for pu_value = [pu_a, pu_b]
        if ~isnan(pu_value) && ~(isfinite(pu_value) && ismember(pu_value, [0, 1]))
            error('parse_matgas_instance:PerUnitFlag', ...
                'mgc.is_per_unit/mgc.per_unit must be 0 or 1.');
        end
    end
    if ~isnan(pu_a) && ~isnan(pu_b) && logical(pu_a) ~= logical(pu_b)
        error('parse_matgas_instance:PerUnitConflict', ...
            'mgc.is_per_unit and mgc.per_unit disagree.');
    elseif ~isnan(pu_a)
        is_per_unit = logical(pu_a);
    elseif ~isnan(pu_b)
        is_per_unit = logical(pu_b);
    else
        is_per_unit = false;
    end

    gas_sg = matgas_scalar_number(clean_text, 'gas_specific_gravity', true, NaN);
    if ~(isfinite(gas_sg) && gas_sg > 0)
        error('parse_matgas_instance:SpecificGravity', ...
            'mgc.gas_specific_gravity must be positive and finite.');
    end
    Tk_raw = matgas_scalar_number(clean_text, 'temperature', true, NaN);
    % MatGas specifies temperature in kelvin for both SI and USC cases.
    Tk = Tk_raw;
    if ~(isfinite(Tk) && Tk > 0)
        error('parse_matgas_instance:Temperature', 'Gas temperature must be positive.');
    end

    z_file = matgas_scalar_number(clean_text, 'compressibility_factor', true, NaN);
    if isempty(opts.matgas_z_override)
        z = z_file;
    else
        z = opts.matgas_z_override;
    end
    if ~(isscalar(z) && isfinite(z) && z > 0)
        error('parse_matgas_instance:Compressibility', ...
            'Compressibility factor (or matgas_z_override) must be positive.');
    end

    Ru = matgas_scalar_number(clean_text, 'R', false, 8.314462618);
    Mkgpmol = matgas_scalar_number(clean_text, 'gas_molar_mass', false, 0.02896 * gas_sg);
    if ~(isfinite(Ru) && Ru > 0 && isfinite(Mkgpmol) && Mkgpmol > 0)
        error('parse_matgas_instance:GasConstants', 'R and gas_molar_mass must be positive.');
    end
    Rs = Ru / Mkgpmol;

    % Same standard-density estimate used by GasModels.  It is used only for
    % MMSCFD conversion and metadata; pipe physics use sound speed/Rs,z,T.
    rho0 = 101325 * (0.02896 * gas_sg) / (Ru * Tk);

    base_pressure_raw = matgas_scalar_number(clean_text, 'base_pressure', false, NaN);
    base_length_raw = matgas_scalar_number(clean_text, 'base_length', false, NaN);
    base_flow_raw = matgas_scalar_number(clean_text, 'base_flow', false, NaN);

    if is_per_unit && any(isnan([base_pressure_raw, base_length_raw]))
        error('parse_matgas_instance:MissingBase', ...
            'Per-unit MatGas data require base_pressure and base_length.');
    end

    base_pressure_pa = matgas_pressure_to_pa(base_pressure_raw, units);
    base_length_m = matgas_length_to_m(base_length_raw, units);
    if is_per_unit && ~(isfinite(base_pressure_pa) && base_pressure_pa > 0 && ...
            isfinite(base_length_m) && base_length_m > 0)
        error('parse_matgas_instance:BaseValue', ...
            'Per-unit base_pressure and base_length must be positive and finite.');
    end

    sound_speed_raw = matgas_scalar_number(clean_text, 'sound_speed', false, NaN);
    if ~isempty(opts.matgas_z_override)
        % An explicit z override is intended to alter the physical model.
        sound_speed = sqrt(Rs * z * Tk);
    elseif isfinite(sound_speed_raw)
        % MatGas specifies sound speed in m/s for both SI and USC cases.
        sound_speed = sound_speed_raw;
    else
        sound_speed = sqrt(Rs * z * Tk);
    end
    if ~(isfinite(sound_speed) && sound_speed > 0)
        error('parse_matgas_instance:SoundSpeed', 'Sound speed must be positive.');
    end
    sound_speed_from_thermo = sqrt(Rs * z * Tk);
    if is_per_unit && isnan(base_flow_raw)
        % GasModels' fallback when an explicit base flow is absent.
        base_flow_kgps = base_pressure_pa / sound_speed;
    else
        base_flow_kgps = matgas_flow_to_kgps_dimensional(base_flow_raw, units, rho0);
    end
    if is_per_unit && ~(isfinite(base_flow_kgps) && base_flow_kgps > 0)
        error('parse_matgas_instance:BaseValue', ...
            'Per-unit base_flow must be positive and finite (or derivable).');
    end

    % ------------------------------------------------------------------
    % Literal tables
    % ------------------------------------------------------------------
    junction_rows   = matgas_table(clean_text, 'junction', true);
    pipe_rows       = matgas_table(clean_text, 'pipe', false);
    compressor_rows = matgas_table(clean_text, 'compressor', false);
    short_rows      = matgas_table(clean_text, 'short_pipe', false);
    valve_rows      = matgas_table(clean_text, 'valve', false);
    regulator_rows  = matgas_table(clean_text, 'regulator', false);
    resistor_rows   = matgas_table(clean_text, 'resistor', false);
    loss_rows       = matgas_table(clean_text, 'loss_resistor', false);
    receipt_rows    = matgas_table(clean_text, 'receipt', false);
    delivery_rows   = matgas_table(clean_text, 'delivery', false);
    transfer_rows   = matgas_table(clean_text, 'transfer', false);
    storage_rows    = matgas_table(clean_text, 'storage', false);
    ne_pipe_rows    = matgas_table(clean_text, 'ne_pipe', false);
    ne_comp_rows    = matgas_table(clean_text, 'ne_compressor', false);
    delivery_ext    = matgas_table(clean_text, 'delivery_data', false);
    regulator_ext   = matgas_table(clean_text, 'regulator_data', false);

    matgas_validate_unique_ids(pipe_rows, 'pipe');
    matgas_validate_unique_ids(compressor_rows, 'compressor');
    matgas_validate_unique_ids(short_rows, 'short_pipe');
    matgas_validate_unique_ids(valve_rows, 'valve');
    matgas_validate_unique_ids(regulator_rows, 'regulator');
    matgas_validate_unique_ids(receipt_rows, 'receipt');
    matgas_validate_unique_ids(delivery_rows, 'delivery');
    matgas_validate_unique_ids(transfer_rows, 'transfer');

    assigned_extensions = matgas_assigned_extension_fields(clean_text);
    known_extensions = {'delivery_data', 'regulator_data'};
    unknown_extensions = setdiff(assigned_extensions, known_extensions, 'stable');
    nonempty_unknown_extensions = {};
    for k = 1:numel(unknown_extensions)
        ext_rows = matgas_table(clean_text, unknown_extensions{k}, false);
        if ~isempty(ext_rows)
            nonempty_unknown_extensions{end+1} = unknown_extensions{k}; %#ok<AGROW>
        end
    end
    if ~isempty(nonempty_unknown_extensions) && opts.matgas_error_on_unknown_extensions
        error('parse_matgas_instance:UnsupportedExtension', ...
            ['Unsupported nonempty MatGas extension table(s): %s. Refusing to ', ...
             'silently ignore model data.'], strjoin(nonempty_unknown_extensions, ', '));
    end

    unsupported = {};
    if matgas_has_active_rows(resistor_rows, 6, 'resistor'), unsupported{end+1} = 'resistor'; end %#ok<AGROW>
    if matgas_has_active_rows(loss_rows, 5, 'loss_resistor'), unsupported{end+1} = 'loss_resistor'; end %#ok<AGROW>
    if matgas_has_active_rows(storage_rows, 12, 'storage'), unsupported{end+1} = 'storage'; end %#ok<AGROW>
    if matgas_has_active_rows(ne_pipe_rows, 9, 'ne_pipe'), unsupported{end+1} = 'ne_pipe'; end %#ok<AGROW>
    if matgas_has_active_rows(ne_comp_rows, 13, 'ne_compressor'), unsupported{end+1} = 'ne_compressor'; end %#ok<AGROW>
    if ~isempty(unsupported) && opts.matgas_error_on_unsupported
        error('parse_matgas_instance:UnsupportedComponent', ...
            ['Active/useful rows of unsupported MatGas table(s) were found: %s. ', ...
             'The current physical model has no corresponding constraints.'], ...
            strjoin(unsupported, ', '));
    end

    parse_warnings = {};
    if ~isempty(unsupported)
        parse_warnings{end+1} = sprintf('Ignored unsupported table(s): %s.', strjoin(unsupported, ', ')); %#ok<AGROW>
    end
    if ~isempty(nonempty_unknown_extensions)
        parse_warnings{end+1} = sprintf('Ignored extension table(s): %s.', ...
            strjoin(nonempty_unknown_extensions, ', ')); %#ok<AGROW>
    end

    % ------------------------------------------------------------------
    % Junctions
    % ------------------------------------------------------------------
    n_all = numel(junction_rows);
    all_node_ids = cell(n_all, 1);
    all_node_keys = cell(n_all, 1);
    all_status = false(n_all, 1);
    all_pmin = nan(n_all, 1);
    all_pmax = nan(n_all, 1);
    all_pnom = nan(n_all, 1);
    all_jtype = zeros(n_all, 1);

    seen_all_nodes = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    for r = 1:n_all
        matgas_require_cols(junction_rows{r}, 6, 'junction', r);
        [all_node_ids{r}, all_node_keys{r}] = matgas_id_token(junction_rows{r}{1}, 'junction', r, 1);
        if isKey(seen_all_nodes, all_node_keys{r})
            error('parse_matgas_instance:DuplicateNode', ...
                'Duplicate junction id %s.', all_node_ids{r});
        end
        seen_all_nodes(all_node_keys{r}) = true;

        pmin0 = matgas_row_number(junction_rows{r}, 2, 'junction', r);
        pmax0 = matgas_row_number(junction_rows{r}, 3, 'junction', r);
        pnom0 = matgas_row_number(junction_rows{r}, 4, 'junction', r);
        all_pmin(r) = matgas_pressure_to_bar(pmin0, units, is_per_unit, base_pressure_pa);
        all_pmax(r) = matgas_pressure_to_bar(pmax0, units, is_per_unit, base_pressure_pa);
        all_pnom(r) = matgas_pressure_to_bar(pnom0, units, is_per_unit, base_pressure_pa);
        all_jtype(r) = matgas_row_integer(junction_rows{r}, 5, 'junction', r);
        all_status(r) = matgas_row_integer(junction_rows{r}, 6, 'junction', r) ~= 0;

        % MatGas uses p_nominal as a reference/reporting value at ordinary
        % junctions, and some official cases place it outside [p_min,p_max].
        % Only reference (type-1) junctions require an in-range nominal
        % pressure; that check is performed after component limits are
        % intersected below.
        if ~(isfinite(all_pmin(r)) && isfinite(all_pmax(r)) && isfinite(all_pnom(r)) && ...
                all_pmin(r) >= 0 && all_pmin(r) <= all_pmax(r))
            error('parse_matgas_instance:PressureBounds', ...
                'Invalid pressure bounds/nominal pressure at junction %s.', all_node_ids{r});
        end
    end

    active_rows = find(all_status);
    if isempty(active_rows)
        error('parse_matgas_instance:NoNodes', 'No active junctions were found.');
    end
    node_ids = all_node_ids(active_rows);
    node_keys = all_node_keys(active_rows);
    pmin_bar = all_pmin(active_rows);
    pmax_bar = all_pmax(active_rows);
    pnom_bar = all_pnom(active_rows);
    junction_type = all_jtype(active_rows);

    N = numel(node_ids);
    node_map = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for i = 1:N
        node_map(node_keys{i}) = i;
    end

    % Intersect junction bounds with active link/compressor engineering
    % pressure limits before fixing any slack pressure.
    for r = 1:numel(pipe_rows)
        row = pipe_rows{r};
        matgas_require_cols(row, 9, 'pipe', r);
        if matgas_row_integer(row, 9, 'pipe', r) == 0, continue; end
        [oid, ~] = matgas_id_token(row{1}, 'pipe', r, 1);
        i = matgas_endpoint(row{2}, node_map, 'pipe', oid, 'from');
        j = matgas_endpoint(row{3}, node_map, 'pipe', oid, 'to');
        link_min = matgas_pressure_to_bar(matgas_row_number(row, 7, 'pipe', r), ...
            units, is_per_unit, base_pressure_pa);
        link_max = matgas_pressure_to_bar(matgas_row_number(row, 8, 'pipe', r), ...
            units, is_per_unit, base_pressure_pa);
        if ~(isfinite(link_min) && isfinite(link_max) && link_min >= 0 && link_min <= link_max)
            error('parse_matgas_instance:PipePressure', ...
                'Invalid pressure limits for pipe %s.', oid);
        end
        pmin_bar([i,j]) = max(pmin_bar([i,j]), link_min);
        pmax_bar([i,j]) = min(pmax_bar([i,j]), link_max);
    end
    for r = 1:numel(compressor_rows)
        row = compressor_rows{r};
        matgas_require_cols(row, 13, 'compressor', r);
        if matgas_row_integer(row, 13, 'compressor', r) == 0, continue; end
        [oid, ~] = matgas_id_token(row{1}, 'compressor', r, 1);
        i = matgas_endpoint(row{2}, node_map, 'compressor', oid, 'from');
        j = matgas_endpoint(row{3}, node_map, 'compressor', oid, 'to');
        pin_min = matgas_pressure_to_bar(matgas_row_number(row, 9, 'compressor', r), ...
            units, is_per_unit, base_pressure_pa);
        pin_max = matgas_pressure_to_bar(matgas_row_number(row, 10, 'compressor', r), ...
            units, is_per_unit, base_pressure_pa);
        pout_min = matgas_pressure_to_bar(matgas_row_number(row, 11, 'compressor', r), ...
            units, is_per_unit, base_pressure_pa);
        pout_max = matgas_pressure_to_bar(matgas_row_number(row, 12, 'compressor', r), ...
            units, is_per_unit, base_pressure_pa);
        if any(~isfinite([pin_min,pin_max,pout_min,pout_max])) || ...
                pin_min < 0 || pout_min < 0 || pin_min > pin_max || pout_min > pout_max
            error('parse_matgas_instance:CompressorPressure', ...
                'Invalid inlet/outlet pressure limits for compressor %s.', oid);
        end
        pmin_bar(i) = max(pmin_bar(i), pin_min);
        pmax_bar(i) = min(pmax_bar(i), pin_max);
        pmin_bar(j) = max(pmin_bar(j), pout_min);
        pmax_bar(j) = min(pmax_bar(j), pout_max);
    end
    for i = 1:N
        if pmin_bar(i) > pmax_bar(i)
            error('parse_matgas_instance:PressureIntersection', ...
                'Component pressure limits conflict at junction %s.', node_ids{i});
        end
    end
    if opts.matgas_fix_slack_pressure
        slack = junction_type == 1;
        if any(pnom_bar(slack) < pmin_bar(slack) | pnom_bar(slack) > pmax_bar(slack))
            error('parse_matgas_instance:SlackPressure', ...
                'A slack junction nominal pressure violates an incident component limit.');
        end
        pmin_bar(slack) = pnom_bar(slack);
        pmax_bar(slack) = pnom_bar(slack);
    end

    % ------------------------------------------------------------------
    % Receipts, deliveries, and transfers
    % ------------------------------------------------------------------
    receipt = matgas_parse_receipts(receipt_rows, node_map, units, is_per_unit, ...
        base_flow_kgps, rho0, opts.matgas_supply_target);
    delivery = matgas_parse_deliveries(delivery_rows, node_map, units, is_per_unit, ...
        base_flow_kgps, rho0, opts.matgas_demand_target);

    if ~isempty(delivery_ext)
        if numel(delivery_ext) ~= numel(delivery_rows)
            error('parse_matgas_instance:DeliveryExtension', ...
                'mgc.delivery_data must have one row per mgc.delivery row.');
        end
        priorities_all = nan(numel(delivery_rows), 1);
        for r = 1:numel(delivery_ext)
            if numel(delivery_ext{r}) ~= 1
                error('parse_matgas_instance:DeliveryExtension', ...
                    ['This parser supports the standard one-column ', ...
                     'mgc.delivery_data priority extension only.']);
            end
            priorities_all(r) = matgas_row_number(delivery_ext{r}, 1, 'delivery_data', r);
        end
        delivery.priority = priorities_all(delivery.source_row);
    else
        delivery.priority = opts.weight_default * ones(numel(delivery.id), 1);
    end
    if ~opts.matgas_use_priorities
        delivery.priority(:) = opts.weight_default;
    end
    if any(~isfinite(delivery.priority) | delivery.priority < 0)
        error('parse_matgas_instance:Priority', ...
            'Delivery priorities must be finite and nonnegative.');
    end

    supply_cap = zeros(N, 1);
    fixed_supply = zeros(N, 1);
    dispatchable_supply = zeros(N, 1);
    for k = 1:numel(receipt.id)
        n = receipt.node_index(k);
        supply_cap(n) = supply_cap(n) + receipt.target(k);
        if receipt.is_dispatchable(k)
            dispatchable_supply(n) = dispatchable_supply(n) + receipt.target(k);
        else
            fixed_supply(n) = fixed_supply(n) + receipt.target(k);
        end
    end

    demand = zeros(N, 1);
    fixed_demand = zeros(N, 1);
    dispatchable_demand = zeros(N, 1);
    weighted_demand = zeros(N, 1);
    weight_values_by_node = cell(N, 1);
    for k = 1:numel(delivery.id)
        n = delivery.node_index(k);
        d = delivery.target(k);
        demand(n) = demand(n) + d;
        weighted_demand(n) = weighted_demand(n) + d * delivery.priority(k);
        if d > 0
            weight_values_by_node{n}(end+1) = delivery.priority(k); %#ok<AGROW>
        end
        if delivery.is_dispatchable(k)
            dispatchable_demand(n) = dispatchable_demand(n) + d;
        else
            fixed_demand(n) = fixed_demand(n) + d;
        end
    end

    transfer = matgas_parse_transfers(transfer_rows, node_map, units, is_per_unit, ...
        base_flow_kgps, rho0, opts.matgas_transfer_mode);
    for k = 1:numel(transfer.id)
        n = transfer.node_index(k);
        v = transfer.nominal(k);
        if v > 0
            demand(n) = demand(n) + v;
            fixed_demand(n) = fixed_demand(n) + v;
            weighted_demand(n) = weighted_demand(n) + v * opts.weight_default;
            weight_values_by_node{n}(end+1) = opts.weight_default; %#ok<AGROW>
        elseif v < 0
            supply_cap(n) = supply_cap(n) - v;
            fixed_supply(n) = fixed_supply(n) - v;
        end
    end

    weights = zeros(N, 1);
    for i = 1:N
        if demand(i) > 0
            weights(i) = weighted_demand(i) / demand(i);
            [has_override, override] = matgas_weight_override(opts.weights, node_ids{i});
            if has_override
                weights(i) = override;
            end
            vals = unique(weight_values_by_node{i});
            if numel(vals) > 1
                parse_warnings{end+1} = sprintf([ ...
                    'Junction %s has deliveries with different priorities; the existing ', ...
                    'node-aggregated model uses their demand-weighted average %.9g.'], ...
                    node_ids{i}, weights(i)); %#ok<AGROW>
            end
        end
    end

    mixed_boundary_nodes = find((fixed_demand > 0 & dispatchable_demand > 0) | ...
                                (fixed_supply > 0 & dispatchable_supply > 0));
    if ~isempty(mixed_boundary_nodes)
        parse_warnings{end+1} = sprintf([ ...
            '%d junction(s) mix fixed and dispatchable boundaries. The raw split is in ', ...
            'data.matgas, but run_gaslib_mgsp aggregates it and therefore does not ', ...
            'exactly reproduce GasModels load-shedding semantics.'], ...
            numel(mixed_boundary_nodes)); %#ok<AGROW>
    end
    if any(fixed_demand > 0) || any(fixed_supply > 0)
        parse_warnings{end+1} = [ ...
            'The existing downstream model relaxes fixed MatGas deliveries into ', ...
            'sheddable demand and fixed receipts into 0-to-cap supply. Raw fixed ', ...
            'amounts are retained in data.matgas.']; %#ok<AGROW>
    end
    if any(receipt.minimum(receipt.is_dispatchable) > 0) || ...
            any(delivery.minimum(delivery.is_dispatchable) > 0)
        parse_warnings{end+1} = [ ...
            'Positive dispatchable boundary minima are retained in data.matgas but ', ...
            'are not enforced by the existing downstream bounds builder.']; %#ok<AGROW>
    end

    scn_entry = supply_cap > 0;
    scn_exit = demand > 0;
    node_types = repmat({'innode'}, N, 1);
    for i = 1:N
        if scn_entry(i) && scn_exit(i)
            node_types{i} = 'source_sink';
        elseif scn_entry(i)
            node_types{i} = 'source';
        elseif scn_exit(i)
            node_types{i} = 'sink';
        elseif junction_type(i) == 1
            node_types{i} = 'slack';
        end
    end

    % Bound used for zero-loss components, which have no native capacity.
    if isempty(opts.matgas_zero_loss_flow_bound)
        finite_candidates = [sum(supply_cap), sum(demand), base_flow_kgps, 1];
        finite_candidates = finite_candidates(isfinite(finite_candidates) & finite_candidates > 0);
        zero_loss_cap = max(finite_candidates);
    else
        zero_loss_cap = opts.matgas_zero_loss_flow_bound;
    end
    if ~(isscalar(zero_loss_cap) && isfinite(zero_loss_cap) && zero_loss_cap > 0)
        error('parse_matgas_instance:FlowBound', ...
            'matgas_zero_loss_flow_bound must be a positive finite scalar.');
    end

    % ------------------------------------------------------------------
    % Arcs
    % ------------------------------------------------------------------
    arc_ids = cell(0, 1);
    arc_original_ids = cell(0, 1);
    arc_types = cell(0, 1);
    arc_subtypes = cell(0, 1);
    from = zeros(0, 1);
    to = zeros(0, 1);
    raw_qmin = zeros(0, 1);
    raw_qmax = zeros(0, 1);
    qmin = zeros(0, 1);
    qmax = zeros(0, 1);
    res_factor = zeros(0, 1);
    crmin = ones(0, 1);
    crmax = ones(0, 1);
    pipe_diameter_m = nan(0, 1);
    pipe_length_m = nan(0, 1);
    pipe_friction = nan(0, 1);

    % Pipes
    for r = 1:numel(pipe_rows)
        row = pipe_rows{r};
        matgas_require_cols(row, 9, 'pipe', r);
        status = matgas_row_integer(row, 9, 'pipe', r);
        if status == 0, continue; end

        [oid, ~] = matgas_id_token(row{1}, 'pipe', r, 1);
        i = matgas_endpoint(row{2}, node_map, 'pipe', oid, 'from');
        j = matgas_endpoint(row{3}, node_map, 'pipe', oid, 'to');
        D = matgas_diameter_to_m(matgas_row_number(row, 4, 'pipe', r), units);
        L = matgas_physical_length(matgas_row_number(row, 5, 'pipe', r), ...
            units, is_per_unit, base_length_m);
        lambda = matgas_row_number(row, 6, 'pipe', r);
        link_pmin = matgas_pressure_to_bar(matgas_row_number(row, 7, 'pipe', r), ...
            units, is_per_unit, base_pressure_pa);
        link_pmax = matgas_pressure_to_bar(matgas_row_number(row, 8, 'pipe', r), ...
            units, is_per_unit, base_pressure_pa);
        if numel(row) >= 10
            bidirectional = matgas_row_integer(row, 10, 'pipe', r) ~= 0;
        else
            bidirectional = true;
        end
        if ~(isfinite(D) && D > 0 && isfinite(L) && L > 0 && ...
                isfinite(lambda) && lambda > 0 && link_pmin >= 0 && link_pmin <= link_pmax)
            error('parse_matgas_instance:PipeData', 'Invalid physical data for pipe %s.', oid);
        end

        K = 16 * lambda * (sound_speed^2) * L / (pi^2 * D^5);
        Kbar = K / 1e10;
        fi_min = max(pmin_bar(i), link_pmin);
        fi_max = min(pmax_bar(i), link_pmax);
        tj_min = max(pmin_bar(j), link_pmin);
        tj_max = min(pmax_bar(j), link_pmax);
        if fi_min > fi_max || tj_min > tj_max
            error('parse_matgas_instance:PipePressure', ...
                'Pipe %s pressure limits conflict with endpoint limits.', oid);
        end
        q_forward = sqrt(max(0, fi_max^2 - tj_min^2) / Kbar);
        q_reverse = sqrt(max(0, tj_max^2 - fi_min^2) / Kbar);
        q_forward = min(q_forward, opts.matgas_max_pipe_flow_bound);
        q_reverse = min(q_reverse, opts.matgas_max_pipe_flow_bound);
        if ~(isfinite(q_forward) && isfinite(q_reverse))
            error('parse_matgas_instance:PipeFlowBound', ...
                'Could not derive finite flow bounds for pipe %s.', oid);
        end

        raw_lo = -q_reverse;
        raw_hi = q_forward;
        if opts.matgas_cap_by_network_flow
            raw_lo = max(raw_lo, -zero_loss_cap);
            raw_hi = min(raw_hi,  zero_loss_cap);
        end
        if opts.passive_bidirectional
            qcap = max(abs([raw_lo, raw_hi]));
            lo = -qcap;
            hi =  qcap;
        elseif bidirectional
            lo = raw_lo;
            hi = raw_hi;
        else
            lo = 0;
            hi = raw_hi;
        end
        [arc_ids, arc_original_ids, arc_types, arc_subtypes, from, to, ...
            raw_qmin, raw_qmax, qmin, qmax, res_factor, crmin, crmax, ...
            pipe_diameter_m, pipe_length_m, pipe_friction] = ...
            matgas_append_arc(arc_ids, arc_original_ids, arc_types, arc_subtypes, ...
            from, to, raw_qmin, raw_qmax, qmin, qmax, res_factor, crmin, crmax, ...
            pipe_diameter_m, pipe_length_m, pipe_friction, ...
            ['pipe:' oid], oid, 'pipe', 'pipe', i, j, raw_lo, raw_hi, lo, hi, ...
            K, 1, 1, D, L, lambda);
    end

    % Short pipes
    for r = 1:numel(short_rows)
        row = short_rows{r};
        matgas_require_cols(row, 4, 'short_pipe', r);
        if matgas_row_integer(row, 4, 'short_pipe', r) == 0, continue; end
        [oid, ~] = matgas_id_token(row{1}, 'short_pipe', r, 1);
        i = matgas_endpoint(row{2}, node_map, 'short_pipe', oid, 'from');
        j = matgas_endpoint(row{3}, node_map, 'short_pipe', oid, 'to');
        bidirectional = true;
        if numel(row) >= 5
            bidirectional = matgas_row_integer(row, 5, 'short_pipe', r) ~= 0;
        end
        raw_lo = -zero_loss_cap; raw_hi = zero_loss_cap;
        if opts.passive_bidirectional || bidirectional, lo = raw_lo; else, lo = 0; end
        hi = raw_hi;
        [arc_ids, arc_original_ids, arc_types, arc_subtypes, from, to, ...
            raw_qmin, raw_qmax, qmin, qmax, res_factor, crmin, crmax, ...
            pipe_diameter_m, pipe_length_m, pipe_friction] = ...
            matgas_append_arc(arc_ids, arc_original_ids, arc_types, arc_subtypes, ...
            from, to, raw_qmin, raw_qmax, qmin, qmax, res_factor, crmin, crmax, ...
            pipe_diameter_m, pipe_length_m, pipe_friction, ...
            ['short_pipe:' oid], oid, 'shortPipe', 'short_pipe', i, j, ...
            raw_lo, raw_hi, lo, hi, 0, 1, 1, NaN, NaN, NaN);
    end

    % Open valves are zero-loss links in this steady simplified model.
    has_active_valves = matgas_has_active_rows(valve_rows, 4, 'valve');
    if has_active_valves && ~opts.matgas_map_valves
        error('parse_matgas_instance:Valve', ...
            'Valve rows found; set opts.matgas_map_valves=true to map open valves to shortPipe arcs.');
    end
    for r = 1:numel(valve_rows)
        row = valve_rows{r};
        matgas_require_cols(row, 4, 'valve', r);
        if matgas_row_integer(row, 4, 'valve', r) == 0, continue; end
        [oid, ~] = matgas_id_token(row{1}, 'valve', r, 1);
        i = matgas_endpoint(row{2}, node_map, 'valve', oid, 'from');
        j = matgas_endpoint(row{3}, node_map, 'valve', oid, 'to');
        raw_lo = -zero_loss_cap; raw_hi = zero_loss_cap;
        if opts.passive_bidirectional, lo = raw_lo; else, lo = 0; end
        hi = raw_hi;
        [arc_ids, arc_original_ids, arc_types, arc_subtypes, from, to, ...
            raw_qmin, raw_qmax, qmin, qmax, res_factor, crmin, crmax, ...
            pipe_diameter_m, pipe_length_m, pipe_friction] = ...
            matgas_append_arc(arc_ids, arc_original_ids, arc_types, arc_subtypes, ...
            from, to, raw_qmin, raw_qmax, qmin, qmax, res_factor, crmin, crmax, ...
            pipe_diameter_m, pipe_length_m, pipe_friction, ...
            ['valve:' oid], oid, 'shortPipe', 'valve', i, j, ...
            raw_lo, raw_hi, lo, hi, 0, 1, 1, NaN, NaN, NaN);
    end
    if has_active_valves
        parse_warnings{end+1} = 'Mapped active valves to zero-pressure-loss shortPipe arcs.'; %#ok<AGROW>
    end

    % Compressors
    for r = 1:numel(compressor_rows)
        row = compressor_rows{r};
        matgas_require_cols(row, 13, 'compressor', r);
        if matgas_row_integer(row, 13, 'compressor', r) == 0, continue; end
        [oid, ~] = matgas_id_token(row{1}, 'compressor', r, 1);
        i = matgas_endpoint(row{2}, node_map, 'compressor', oid, 'from');
        j = matgas_endpoint(row{3}, node_map, 'compressor', oid, 'to');
        rmin = matgas_row_number(row, 4, 'compressor', r);
        rmax = matgas_row_number(row, 5, 'compressor', r);
        raw_lo = matgas_flow_to_kgps(matgas_row_number(row, 7, 'compressor', r), ...
            units, is_per_unit, base_flow_kgps, rho0);
        raw_hi = matgas_flow_to_kgps(matgas_row_number(row, 8, 'compressor', r), ...
            units, is_per_unit, base_flow_kgps, rho0);
        if ~(isfinite(rmin) && rmin > 0 && isfinite(rmax) && rmin <= rmax && ...
                isfinite(raw_lo) && isfinite(raw_hi) && raw_lo <= raw_hi)
            error('parse_matgas_instance:CompressorData', ...
                'Invalid ratio or flow bounds for compressor %s.', oid);
        end
        if opts.matgas_cap_by_network_flow
            raw_lo = max(raw_lo, -zero_loss_cap);
            raw_hi = min(raw_hi,  zero_loss_cap);
        end
        if raw_lo > raw_hi
            error('parse_matgas_instance:CompressorFlowBound', ...
                'Network flow cap conflicts with compressor %s flow bounds.', oid);
        end
        if opts.enforce_compressor_forward, lo = max(0, raw_lo); else, lo = raw_lo; end
        hi = raw_hi;
        if lo > hi
            error('parse_matgas_instance:CompressorDirection', ...
                'Forward-flow enforcement conflicts with compressor %s flow bounds.', oid);
        end
        [arc_ids, arc_original_ids, arc_types, arc_subtypes, from, to, ...
            raw_qmin, raw_qmax, qmin, qmax, res_factor, crmin, crmax, ...
            pipe_diameter_m, pipe_length_m, pipe_friction] = ...
            matgas_append_arc(arc_ids, arc_original_ids, arc_types, arc_subtypes, ...
            from, to, raw_qmin, raw_qmax, qmin, qmax, res_factor, crmin, crmax, ...
            pipe_diameter_m, pipe_length_m, pipe_friction, ...
            ['compressor:' oid], oid, 'compressorStation', 'compressor', i, j, ...
            raw_lo, raw_hi, lo, hi, 0, rmin, rmax, NaN, NaN, NaN);
    end

    % Regulators are pressure-ratio arcs with ratios <= 1.
    has_active_regulators = matgas_has_active_rows(regulator_rows, 8, 'regulator');
    if has_active_regulators && ~opts.matgas_map_regulators
        error('parse_matgas_instance:Regulator', ...
            'Regulator rows found; set opts.matgas_map_regulators=true to map them to ratio arcs.');
    end
    if ~isempty(regulator_ext) && numel(regulator_ext) ~= numel(regulator_rows)
        error('parse_matgas_instance:RegulatorExtension', ...
            'mgc.regulator_data must have one row per mgc.regulator row.');
    end
    for r = 1:numel(regulator_rows)
        row = regulator_rows{r};
        matgas_require_cols(row, 8, 'regulator', r);
        if matgas_row_integer(row, 8, 'regulator', r) == 0, continue; end
        [oid, ~] = matgas_id_token(row{1}, 'regulator', r, 1);
        i = matgas_endpoint(row{2}, node_map, 'regulator', oid, 'from');
        j = matgas_endpoint(row{3}, node_map, 'regulator', oid, 'to');
        rmin = matgas_row_number(row, 4, 'regulator', r);
        rmax = matgas_row_number(row, 5, 'regulator', r);
        raw_lo = matgas_flow_to_kgps(matgas_row_number(row, 6, 'regulator', r), ...
            units, is_per_unit, base_flow_kgps, rho0);
        raw_hi = matgas_flow_to_kgps(matgas_row_number(row, 7, 'regulator', r), ...
            units, is_per_unit, base_flow_kgps, rho0);
        if ~(isfinite(rmin) && rmin >= 0 && isfinite(rmax) && rmin <= rmax && rmax <= 1 + 1e-10)
            error('parse_matgas_instance:RegulatorData', ...
                'Invalid reduction factors for regulator %s.', oid);
        end
        if opts.matgas_cap_by_network_flow
            raw_lo = max(raw_lo, -zero_loss_cap);
            raw_hi = min(raw_hi,  zero_loss_cap);
        end
        if raw_lo > raw_hi
            error('parse_matgas_instance:RegulatorFlowBound', ...
                'Network flow cap conflicts with regulator %s flow bounds.', oid);
        end
        if opts.enforce_compressor_forward, lo = max(0, raw_lo); else, lo = raw_lo; end
        hi = raw_hi;
        if lo > hi
            error('parse_matgas_instance:RegulatorDirection', ...
                'Forward-flow enforcement conflicts with regulator %s flow bounds.', oid);
        end
        [arc_ids, arc_original_ids, arc_types, arc_subtypes, from, to, ...
            raw_qmin, raw_qmax, qmin, qmax, res_factor, crmin, crmax, ...
            pipe_diameter_m, pipe_length_m, pipe_friction] = ...
            matgas_append_arc(arc_ids, arc_original_ids, arc_types, arc_subtypes, ...
            from, to, raw_qmin, raw_qmax, qmin, qmax, res_factor, crmin, crmax, ...
            pipe_diameter_m, pipe_length_m, pipe_friction, ...
            ['regulator:' oid], oid, 'compressorStation', 'regulator', i, j, ...
            raw_lo, raw_hi, lo, hi, 0, rmin, rmax, NaN, NaN, NaN);
    end
    if has_active_regulators
        parse_warnings{end+1} = [ ...
            'Mapped active regulators to compressorStation ratio arcs. This matches ', ...
            'the simplified model for forward flow; reverse regulator operation is not exact.']; %#ok<AGROW>
    end

    M = numel(arc_ids);
    if M == 0
        error('parse_matgas_instance:NoArcs', 'No supported active arcs were found.');
    end

    % ------------------------------------------------------------------
    % Output in run_gaslib_mgsp parser schema
    % ------------------------------------------------------------------
    net_title = matgas_scalar_string(clean_text, 'name', false, '');
    if isempty(net_title)
        net_title = matgas_function_name(clean_text);
    end
    if isempty(net_title)
        [~, net_title] = fileparts(matgas_file);
    end

    data = struct();
    data.net_file = matgas_file;
    data.scn_file = '';
    data.net_title = string(net_title);
    data.scn_id = 'matgas';
    data.N = N;
    data.M = M;
    data.node_ids = node_ids;
    data.node_types = node_types;
    data.arc_ids = arc_ids;
    data.arc_types = arc_types;
    data.from = from;
    data.to = to;
    data.raw_qmin = raw_qmin;
    data.raw_qmax = raw_qmax;
    data.qmin = qmin;
    data.qmax = qmax;
    data.pmin_bar = pmin_bar;
    data.pmax_bar = pmax_bar;
    data.Pa_per_bar = 1e5;
    data.Pa2_per_bar2 = 1e10;
    data.height_m = zeros(N, 1);
    data.node_flowmax_kgps = nan(N, 1);
    data.node_flowmin_kgps = nan(N, 1);
    data.rho0 = rho0;
    data.Tk = Tk;
    data.molar_mass_kg_per_mol = Mkgpmol;
    data.calorific_MJ_per_m3 = NaN;
    data.Rs = Rs;
    data.z = z;
    data.res_factor = res_factor;
    data.res_over_pa2 = res_factor / data.Pa2_per_bar2;
    data.crmin = crmin;
    data.crmax = crmax;
    data.scn_entry = scn_entry;
    data.scn_exit = scn_exit;
    data.supply_cap = supply_cap;
    data.demand = demand;
    data.weights = weights;
    data.pipe_idx = find(strcmp(arc_types, 'pipe'));
    data.short_idx = find(strcmp(arc_types, 'shortPipe'));
    data.comp_idx = find(strcmp(arc_types, 'compressorStation'));

    data.matgas = struct();
    data.matgas.units = units;
    data.matgas.is_per_unit_input = is_per_unit;
    data.matgas.base_pressure_pa = base_pressure_pa;
    data.matgas.base_length_m = base_length_m;
    data.matgas.base_flow_kgps = base_flow_kgps;
    data.matgas.pnom_bar = pnom_bar;
    data.matgas.junction_type = junction_type;
    data.matgas.sound_speed_mps = sound_speed;
    data.matgas.sound_speed_from_thermo_mps = sound_speed_from_thermo;
    data.matgas.arc_original_ids = arc_original_ids;
    data.matgas.arc_subtypes = arc_subtypes;
    data.matgas.pipe_diameter_m = pipe_diameter_m;
    data.matgas.pipe_length_m = pipe_length_m;
    data.matgas.pipe_friction_factor = pipe_friction;
    data.matgas.receipt = receipt;
    data.matgas.delivery = delivery;
    data.matgas.transfer = transfer;
    data.matgas.fixed_supply = fixed_supply;
    data.matgas.dispatchable_supply = dispatchable_supply;
    data.matgas.fixed_demand = fixed_demand;
    data.matgas.dispatchable_demand = dispatchable_demand;
    data.matgas.mixed_boundary_nodes = mixed_boundary_nodes;
    data.matgas.zero_loss_flow_bound_kgps = zero_loss_cap;
    data.matgas.parse_warnings = parse_warnings;

    if opts.verbose
        fprintf('\n=== MatGas parse ===\n');
        fprintf('File      : %s\n', matgas_file);
        fprintf('Network   : %s\n', char(data.net_title));
        fprintf('Units     : %s  (per-unit input=%d; returned in SI/bar)\n', units, is_per_unit);
        fprintf('Nodes     : %d\n', N);
        fprintf('Arcs      : %d  (pipes=%d, short/valve=%d, compressor/regulator=%d)\n', ...
            M, numel(data.pipe_idx), numel(data.short_idx), numel(data.comp_idx));
        fprintf('Supply cap: %.9g kg/s\n', sum(supply_cap));
        fprintf('Demand    : %.9g kg/s\n', sum(demand));
        for k = 1:numel(parse_warnings)
            warning('parse_matgas_instance:Approximation', '%s', parse_warnings{k});
        end
    end
end

% ======================================================================
% Boundary table parsers
% ======================================================================

function receipt = matgas_parse_receipts(rows, node_map, units, is_pu, base_flow, rho0, mode)
    receipt = matgas_empty_boundary('receipt');
    for r = 1:numel(rows)
        row = rows{r};
        matgas_require_cols(row, 7, 'receipt', r);
        status = matgas_row_integer(row, 7, 'receipt', r);
        if status == 0, continue; end
        [id, ~] = matgas_id_token(row{1}, 'receipt', r, 1);
        n = matgas_endpoint(row{2}, node_map, 'receipt', id, 'junction');
        vmin = matgas_flow_to_kgps(matgas_row_number(row, 3, 'receipt', r), units, is_pu, base_flow, rho0);
        vmax = matgas_flow_to_kgps(matgas_row_number(row, 4, 'receipt', r), units, is_pu, base_flow, rho0);
        vnom = matgas_flow_to_kgps(matgas_row_number(row, 5, 'receipt', r), units, is_pu, base_flow, rho0);
        dispflag = matgas_row_integer(row, 6, 'receipt', r) ~= 0;
        if any(~isfinite([vmin,vmax,vnom])) || vmin < 0 || ...
                vmin > vmax || vnom < vmin - 1e-9 || vnom > vmax + 1e-9
            error('parse_matgas_instance:ReceiptBounds', 'Invalid bounds for receipt %s.', id);
        end
        target = matgas_boundary_target(vnom, vmax, dispflag, mode);
        if target < -1e-10
            error('parse_matgas_instance:ReceiptSign', 'Receipt %s has a negative target.', id);
        end
        receipt.id{end+1,1} = id;
        receipt.node_index(end+1,1) = n;
        receipt.minimum(end+1,1) = vmin;
        receipt.maximum(end+1,1) = vmax;
        receipt.nominal(end+1,1) = vnom;
        receipt.is_dispatchable(end+1,1) = dispflag;
        receipt.target(end+1,1) = max(0, target);
        receipt.source_row(end+1,1) = r;
    end
end

function delivery = matgas_parse_deliveries(rows, node_map, units, is_pu, base_flow, rho0, mode)
    delivery = matgas_empty_boundary('delivery');
    for r = 1:numel(rows)
        row = rows{r};
        matgas_require_cols(row, 7, 'delivery', r);
        status = matgas_row_integer(row, 7, 'delivery', r);
        if status == 0, continue; end
        [id, ~] = matgas_id_token(row{1}, 'delivery', r, 1);
        n = matgas_endpoint(row{2}, node_map, 'delivery', id, 'junction');
        vmin = matgas_flow_to_kgps(matgas_row_number(row, 3, 'delivery', r), units, is_pu, base_flow, rho0);
        vmax = matgas_flow_to_kgps(matgas_row_number(row, 4, 'delivery', r), units, is_pu, base_flow, rho0);
        vnom = matgas_flow_to_kgps(matgas_row_number(row, 5, 'delivery', r), units, is_pu, base_flow, rho0);
        dispflag = matgas_row_integer(row, 6, 'delivery', r) ~= 0;
        if any(~isfinite([vmin,vmax,vnom])) || vmin < 0 || ...
                vmin > vmax || vnom < vmin - 1e-9 || vnom > vmax + 1e-9
            error('parse_matgas_instance:DeliveryBounds', 'Invalid bounds for delivery %s.', id);
        end
        target = matgas_boundary_target(vnom, vmax, dispflag, mode);
        if target < -1e-10
            error('parse_matgas_instance:DeliverySign', 'Delivery %s has a negative target.', id);
        end
        delivery.id{end+1,1} = id;
        delivery.node_index(end+1,1) = n;
        delivery.minimum(end+1,1) = vmin;
        delivery.maximum(end+1,1) = vmax;
        delivery.nominal(end+1,1) = vnom;
        delivery.is_dispatchable(end+1,1) = dispflag;
        delivery.target(end+1,1) = max(0, target);
        delivery.source_row(end+1,1) = r;
    end
end

function transfer = matgas_parse_transfers(rows, node_map, units, is_pu, base_flow, rho0, mode)
    transfer = matgas_empty_boundary('transfer');
    transfer.nominal = zeros(0,1);
    for r = 1:numel(rows)
        row = rows{r};
        matgas_require_cols(row, 7, 'transfer', r);
        if matgas_row_integer(row, 7, 'transfer', r) == 0, continue; end
        [id, ~] = matgas_id_token(row{1}, 'transfer', r, 1);
        n = matgas_endpoint(row{2}, node_map, 'transfer', id, 'junction');
        vmin = matgas_flow_to_kgps(matgas_row_number(row, 3, 'transfer', r), units, is_pu, base_flow, rho0);
        vmax = matgas_flow_to_kgps(matgas_row_number(row, 4, 'transfer', r), units, is_pu, base_flow, rho0);
        vnom = matgas_flow_to_kgps(matgas_row_number(row, 5, 'transfer', r), units, is_pu, base_flow, rho0);
        dispflag = matgas_row_integer(row, 6, 'transfer', r) ~= 0;
        if any(~isfinite([vmin,vmax,vnom])) || ...
                vmin > vmax || vnom < vmin - 1e-9 || vnom > vmax + 1e-9
            error('parse_matgas_instance:TransferBounds', 'Invalid bounds for transfer %s.', id);
        end
        if dispflag
            switch mode
                case 'error'
                    if abs(vmin - vnom) > 1e-10 || abs(vmax - vnom) > 1e-10
                        error('parse_matgas_instance:DispatchableTransfer', ...
                            ['Transfer %s is dispatchable. The current solver has no signed, ', ...
                             'mutually-exclusive transfer variable; use transfer_mode ', ...
                             '''nominal'' or ''ignore'' only if that approximation is intended.'], id);
                    end
                case 'ignore'
                    continue;
                case 'nominal'
                    % handled below
                otherwise
                    error('parse_matgas_instance:TransferMode', 'Unknown transfer mode %s.', mode);
            end
        end
        transfer.id{end+1,1} = id;
        transfer.node_index(end+1,1) = n;
        transfer.minimum(end+1,1) = vmin;
        transfer.maximum(end+1,1) = vmax;
        transfer.nominal(end+1,1) = vnom;
        transfer.is_dispatchable(end+1,1) = dispflag;
        transfer.target(end+1,1) = abs(vnom);
        transfer.source_row(end+1,1) = r;
    end
end

function s = matgas_empty_boundary(~)
    s = struct();
    s.id = cell(0,1);
    s.node_index = zeros(0,1);
    s.minimum = zeros(0,1);
    s.maximum = zeros(0,1);
    s.nominal = zeros(0,1);
    s.is_dispatchable = false(0,1);
    s.target = zeros(0,1);
    s.source_row = zeros(0,1);
end

function value = matgas_boundary_target(nominal, maximum, is_dispatchable, mode)
    switch mode
        case 'auto'
            if is_dispatchable, value = maximum; else, value = nominal; end
        case 'nominal'
            value = nominal;
        case 'maximum'
            value = maximum;
        otherwise
            error('parse_matgas_instance:BoundaryMode', 'Unknown boundary target mode ''%s''.', mode);
    end
end

% ======================================================================
% Safe MatGas literal lexer
% ======================================================================

function text = matgas_normalize_text(text)
    if ~isempty(text) && double(text(1)) == 65279
        text = text(2:end); % UTF-8 BOM after fileread decoding
    end
    text = strrep(text, sprintf('\r\n'), sprintf('\n'));
    text = strrep(text, sprintf('\r'), sprintf('\n'));
end

function clean = matgas_strip_comments(text)
    % Remove MATLAB comments outside quoted character vectors.  Preserve
    % newlines so that table rows remain delimited.
    out = repmat(' ', 1, numel(text));
    nout = 0;
    in_quote = false;
    i = 1;
    while i <= numel(text)
        ch = text(i);
        if in_quote
            nout = nout + 1; out(nout) = ch;
            if ch == ''''
                if i < numel(text) && text(i+1) == ''''
                    i = i + 1;
                    nout = nout + 1; out(nout) = text(i);
                else
                    in_quote = false;
                end
            end
        else
            if ch == ''''
                in_quote = true;
                nout = nout + 1; out(nout) = ch;
            elseif ch == '%'
                while i <= numel(text) && text(i) ~= sprintf('\n')
                    i = i + 1;
                end
                if i <= numel(text)
                    nout = nout + 1; out(nout) = sprintf('\n');
                end
            else
                nout = nout + 1; out(nout) = ch;
            end
        end
        i = i + 1;
    end
    if in_quote
        error('parse_matgas_instance:Quote', 'Unterminated quoted string in MatGas file.');
    end
    clean = out(1:nout);
    clean = regexprep(clean, '\.\.\.[ \t]*\n', ' ');
end

function rows = matgas_table(text, field, required)
    [start_pos, value_pos] = matgas_assignment(text, field);
    if isempty(start_pos)
        if required
            error('parse_matgas_instance:MissingTable', 'Missing mgc.%s table.', field);
        end
        rows = {};
        return;
    end
    if text(value_pos) ~= '['
        error('parse_matgas_instance:TableSyntax', 'mgc.%s must be assigned a literal [...].', field);
    end
    close_pos = matgas_matching_bracket(text, value_pos);
    body = text(value_pos+1:close_pos-1);
    row_text = matgas_split_rows(body);
    rows = cell(0,1);
    for r = 1:numel(row_text)
        tokens = matgas_tokenize_row(row_text{r});
        if ~isempty(tokens)
            rows{end+1,1} = tokens; %#ok<AGROW>
        end
    end
end

function fields = matgas_assigned_extension_fields(text)
    toks = regexp(text, ...
        '(?m)^[ \t]*mgc\.([A-Za-z][A-Za-z0-9_]*_data)[ \t]*=[ \t]*\[', ...
        'tokens');
    fields = cell(0,1);
    for k = 1:numel(toks)
        fields{end+1,1} = toks{k}{1}; %#ok<AGROW>
    end
    fields = unique(fields, 'stable');
end

function [start_pos, value_pos] = matgas_assignment(text, field)
    escaped = regexptranslate('escape', field);
    pattern = ['(?m)^[ \t]*mgc\.' escaped '[ \t]*=[ \t]*'];
    [starts, ends] = regexp(text, pattern, 'start', 'end');
    if isempty(starts)
        start_pos = []; value_pos = [];
        return;
    end
    if numel(starts) > 1
        error('parse_matgas_instance:DuplicateAssignment', ...
            'mgc.%s is assigned more than once.', field);
    end
    start_pos = starts(1);
    value_pos = ends(1) + 1;
    while value_pos <= numel(text) && ismember(text(value_pos), sprintf(' \t\n'))
        value_pos = value_pos + 1;
    end
    if value_pos > numel(text)
        error('parse_matgas_instance:Assignment', 'Missing value for mgc.%s.', field);
    end
end

function close_pos = matgas_matching_bracket(text, open_pos)
    depth = 0;
    in_quote = false;
    i = open_pos;
    while i <= numel(text)
        ch = text(i);
        if in_quote
            if ch == ''''
                if i < numel(text) && text(i+1) == ''''
                    i = i + 1;
                else
                    in_quote = false;
                end
            end
        else
            if ch == ''''
                in_quote = true;
            elseif ch == '['
                depth = depth + 1;
            elseif ch == ']'
                depth = depth - 1;
                if depth == 0
                    close_pos = i;
                    return;
                elseif depth < 0
                    break;
                end
            end
        end
        i = i + 1;
    end
    error('parse_matgas_instance:Bracket', 'Unterminated table beginning at character %d.', open_pos);
end

function rows = matgas_split_rows(body)
    rows = cell(0,1);
    buf = repmat(' ', 1, max(1, numel(body)));
    nb = 0;
    in_quote = false;
    i = 1;
    while i <= numel(body)
        ch = body(i);
        if in_quote
            nb = nb + 1; buf(nb) = ch;
            if ch == ''''
                if i < numel(body) && body(i+1) == ''''
                    i = i + 1; nb = nb + 1; buf(nb) = body(i);
                else
                    in_quote = false;
                end
            end
        else
            if ch == ''''
                in_quote = true; nb = nb + 1; buf(nb) = ch;
            elseif ch == ';' || ch == sprintf('\n')
                piece = strtrim(buf(1:nb));
                if ~isempty(piece), rows{end+1,1} = piece; end %#ok<AGROW>
                nb = 0;
            else
                nb = nb + 1; buf(nb) = ch;
            end
        end
        i = i + 1;
    end
    piece = strtrim(buf(1:nb));
    if ~isempty(piece), rows{end+1,1} = piece; end
end

function tokens = matgas_tokenize_row(row)
    tokens = cell(0,1);
    i = 1;
    while i <= numel(row)
        while i <= numel(row) && (isspace(row(i)) || row(i) == ',')
            i = i + 1;
        end
        if i > numel(row), break; end
        if row(i) == ''''
            start = i;
            i = i + 1;
            closed = false;
            while i <= numel(row)
                if row(i) == ''''
                    if i < numel(row) && row(i+1) == ''''
                        i = i + 2;
                    else
                        i = i + 1;
                        closed = true;
                        break;
                    end
                else
                    i = i + 1;
                end
            end
            if ~closed
                error('parse_matgas_instance:Quote', 'Unterminated string in table row: %s', row);
            end
            tokens{end+1,1} = row(start:i-1); %#ok<AGROW>
        else
            start = i;
            while i <= numel(row) && ~isspace(row(i)) && row(i) ~= ','
                i = i + 1;
            end
            tokens{end+1,1} = row(start:i-1); %#ok<AGROW>
        end
    end
end

function value = matgas_scalar_number(text, field, required, default_value)
    token = matgas_scalar_token(text, field, required);
    if isempty(token)
        value = default_value;
    else
        value = matgas_numeric_literal(token, ['mgc.' field]);
    end
end

function value = matgas_scalar_string(text, field, required, default_value)
    token = matgas_scalar_token(text, field, required);
    if isempty(token)
        value = default_value;
    elseif token(1) == '''' && token(end) == ''''
        value = strrep(token(2:end-1), '''''', '''');
    else
        error('parse_matgas_instance:StringLiteral', 'mgc.%s must be a quoted string.', field);
    end
end

function token = matgas_scalar_token(text, field, required)
    [start_pos, value_pos] = matgas_assignment(text, field); %#ok<ASGLU>
    if isempty(start_pos)
        if required
            error('parse_matgas_instance:MissingScalar', 'Missing mgc.%s.', field);
        end
        token = '';
        return;
    end
    if text(value_pos) == '['
        error('parse_matgas_instance:ScalarSyntax', 'mgc.%s must be a scalar.', field);
    end
    in_quote = false;
    i = value_pos;
    while i <= numel(text)
        ch = text(i);
        if in_quote
            if ch == ''''
                if i < numel(text) && text(i+1) == ''''
                    i = i + 1;
                else
                    in_quote = false;
                end
            end
        else
            if ch == ''''
                in_quote = true;
            elseif ch == ';' || ch == sprintf('\n')
                break;
            end
        end
        i = i + 1;
    end
    token = strtrim(text(value_pos:i-1));
    if isempty(token)
        error('parse_matgas_instance:ScalarSyntax', 'Empty value for mgc.%s.', field);
    end
end

function value = matgas_numeric_literal(token, context)
    token = strtrim(token);
    lo = lower(token);
    if strcmp(lo, 'true'), value = 1; return; end
    if strcmp(lo, 'false'), value = 0; return; end
    if strcmp(lo, 'inf') || strcmp(lo, '+inf'), value = Inf; return; end
    if strcmp(lo, '-inf'), value = -Inf; return; end
    if strcmp(lo, 'nan') || strcmp(lo, '+nan') || strcmp(lo, '-nan'), value = NaN; return; end
    pattern = '^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eEdD][+-]?\d+)?$';
    if isempty(regexp(token, pattern, 'once'))
        error('parse_matgas_instance:NumericLiteral', ...
            'Expected a numeric literal for %s; found ''%s''.', context, token);
    end
    token(token == 'd' | token == 'D') = 'e';
    value = str2double(token);
    if isnan(value)
        error('parse_matgas_instance:NumericLiteral', 'Could not parse numeric literal %s.', token);
    end
end

% ======================================================================
% Validation, units, and small utilities
% ======================================================================

function matgas_require_cols(row, n, table_name, row_number)
    if numel(row) < n
        error('parse_matgas_instance:ShortRow', ...
            'mgc.%s row %d has %d columns; at least %d are required.', ...
            table_name, row_number, numel(row), n);
    end
end

function tf = matgas_has_active_rows(rows, status_col, table_name)
    tf = false;
    for r = 1:numel(rows)
        matgas_require_cols(rows{r}, status_col, table_name, r);
        if matgas_row_integer(rows{r}, status_col, table_name, r) ~= 0
            tf = true;
            return;
        end
    end
end

function matgas_validate_unique_ids(rows, table_name)
    seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    for r = 1:numel(rows)
        matgas_require_cols(rows{r}, 1, table_name, r);
        [display_id, key] = matgas_id_token(rows{r}{1}, table_name, r, 1);
        if isKey(seen, key)
            error('parse_matgas_instance:DuplicateId', ...
                'Duplicate id %s in mgc.%s.', display_id, table_name);
        end
        seen(key) = true;
    end
end

function value = matgas_row_number(row, col, table_name, row_number)
    matgas_require_cols(row, col, table_name, row_number);
    token = row{col};
    if token(1) == ''''
        error('parse_matgas_instance:NumericColumn', ...
            'mgc.%s row %d column %d must be numeric.', table_name, row_number, col);
    end
    value = matgas_numeric_literal(token, sprintf('mgc.%s row %d column %d', table_name, row_number, col));
end

function value = matgas_row_integer(row, col, table_name, row_number)
    value = matgas_row_number(row, col, table_name, row_number);
    if ~(isfinite(value) && abs(value - round(value)) <= 1e-9)
        error('parse_matgas_instance:IntegerColumn', ...
            'mgc.%s row %d column %d must be an integer.', table_name, row_number, col);
    end
    value = round(value);
end

function [display_id, key] = matgas_id_token(token, table_name, row_number, col)
    if token(1) == ''''
        display_id = strrep(token(2:end-1), '''''', '''');
        key = ['s:' display_id];
    else
        v = matgas_numeric_literal(token, sprintf('mgc.%s row %d column %d', table_name, row_number, col));
        if ~(isfinite(v) && abs(v - round(v)) <= 1e-9)
            error('parse_matgas_instance:Id', ...
                'mgc.%s row %d has a noninteger numeric id.', table_name, row_number);
        end
        v = round(v);
        display_id = sprintf('%.0f', v);
        key = ['n:' display_id];
    end
    if isempty(display_id)
        error('parse_matgas_instance:Id', 'Empty id in mgc.%s row %d.', table_name, row_number);
    end
end

function n = matgas_endpoint(token, node_map, table_name, component_id, side)
    [display_id, key] = matgas_id_token(token, table_name, 0, 0);
    if ~isKey(node_map, key)
        error('parse_matgas_instance:UnknownNode', ...
            'mgc.%s component %s references inactive/unknown %s junction %s.', ...
            table_name, component_id, side, display_id);
    end
    n = node_map(key);
end

function p_pa = matgas_pressure_to_pa(value, units)
    if isnan(value), p_pa = NaN; return; end
    if strcmp(units, 'si')
        p_pa = value;
    else
        p_pa = value * 6894.757293168;
    end
end

function p_bar = matgas_pressure_to_bar(value, units, is_pu, base_pressure_pa)
    if is_pu
        p_pa = value * base_pressure_pa;
    else
        p_pa = matgas_pressure_to_pa(value, units);
    end
    p_bar = p_pa / 1e5;
end

function L = matgas_length_to_m(value, units)
    if isnan(value), L = NaN; return; end
    if strcmp(units, 'si'), L = value; else, L = value * 1609.64; end
end

function L = matgas_physical_length(value, units, is_pu, base_length_m)
    if is_pu, L = value * base_length_m; else, L = matgas_length_to_m(value, units); end
end

function D = matgas_diameter_to_m(value, units)
    if strcmp(units, 'si'), D = value; else, D = value * 0.0254; end
end

function q = matgas_flow_to_kgps(value, units, is_pu, base_flow_kgps, rho0)
    if is_pu
        q = value * base_flow_kgps;
    else
        q = matgas_flow_to_kgps_dimensional(value, units, rho0);
    end
end

function q = matgas_flow_to_kgps_dimensional(value, units, rho0)
    if isnan(value), q = NaN; return; end
    if strcmp(units, 'si')
        q = value;
    else
        m3_per_s_per_mmscfd = 1e6 * 0.02832 / 86400;
        q = value * m3_per_s_per_mmscfd * rho0;
    end
end

function [found, value] = matgas_weight_override(weights, node_id)
    found = false; value = NaN;
    if isempty(weights) || ~isstruct(weights), return; end
    candidates = {node_id, ['node_' node_id], ['sink_' node_id]};
    try
        candidates{end+1} = matlab.lang.makeValidName(node_id); %#ok<AGROW>
    catch
        % Older MATLAB: node_<id> remains available.
    end
    for k = 1:numel(candidates)
        f = candidates{k};
        if isvarname(f) && isfield(weights, f)
            value = weights.(f);
            if ~(isscalar(value) && isfinite(value) && value >= 0)
                error('parse_matgas_instance:Weight', 'Weight override %s must be a nonnegative scalar.', f);
            end
            found = true;
            return;
        end
    end
end

function name = matgas_function_name(text)
    tok = regexp(text, '(?m)^\s*function\s+\w+\s*=\s*([A-Za-z0-9_.-]+)', 'tokens', 'once');
    if isempty(tok), name = ''; else, name = tok{1}; end
end

function opts = matgas_default_opts(opts)
    opts = matgas_set_default(opts, 'passive_bidirectional', true);
    opts = matgas_set_default(opts, 'enforce_compressor_forward', true);
    opts = matgas_set_default(opts, 'weight_default', 1);
    opts = matgas_set_default(opts, 'weights', struct());
    opts = matgas_set_default(opts, 'verbose', true);
    opts = matgas_set_default(opts, 'matgas_demand_target', 'auto');
    opts = matgas_set_default(opts, 'matgas_supply_target', 'auto');
    opts = matgas_set_default(opts, 'matgas_use_priorities', true);
    opts = matgas_set_default(opts, 'matgas_fix_slack_pressure', true);
    opts = matgas_set_default(opts, 'matgas_z_override', []);
    opts = matgas_set_default(opts, 'matgas_zero_loss_flow_bound', []);
    opts = matgas_set_default(opts, 'matgas_max_pipe_flow_bound', Inf);
    opts = matgas_set_default(opts, 'matgas_cap_by_network_flow', false);
    opts = matgas_set_default(opts, 'matgas_map_valves', true);
    opts = matgas_set_default(opts, 'matgas_map_regulators', true);
    opts = matgas_set_default(opts, 'matgas_transfer_mode', 'error');
    opts = matgas_set_default(opts, 'matgas_error_on_unsupported', true);
    opts = matgas_set_default(opts, 'matgas_error_on_unknown_extensions', true);
    opts.matgas_demand_target = lower(char(opts.matgas_demand_target));
    opts.matgas_supply_target = lower(char(opts.matgas_supply_target));
    opts.matgas_transfer_mode = lower(char(opts.matgas_transfer_mode));
    if ~(isscalar(opts.weight_default) && isfinite(opts.weight_default) && opts.weight_default >= 0)
        error('parse_matgas_instance:Option', 'weight_default must be finite and nonnegative.');
    end
    if ~(isscalar(opts.matgas_max_pipe_flow_bound) && opts.matgas_max_pipe_flow_bound > 0)
        error('parse_matgas_instance:Option', 'matgas_max_pipe_flow_bound must be positive.');
    end
end

function s = matgas_set_default(s, name, value)
    if ~isfield(s, name) || isempty(s.(name))
        s.(name) = value;
    end
end

function varargout = matgas_append_arc(arc_ids, arc_original_ids, arc_types, arc_subtypes, ...
        from, to, raw_qmin, raw_qmax, qmin, qmax, res_factor, crmin, crmax, ...
        pipe_diameter_m, pipe_length_m, pipe_friction, ...
        new_id, original_id, typ, subtype, i, j, raw_lo, raw_hi, lo, hi, K, rmin, rmax, D, L, lambda)
    arc_ids{end+1,1} = new_id;
    arc_original_ids{end+1,1} = original_id;
    arc_types{end+1,1} = typ;
    arc_subtypes{end+1,1} = subtype;
    from(end+1,1) = i;
    to(end+1,1) = j;
    raw_qmin(end+1,1) = raw_lo;
    raw_qmax(end+1,1) = raw_hi;
    qmin(end+1,1) = lo;
    qmax(end+1,1) = hi;
    res_factor(end+1,1) = K;
    crmin(end+1,1) = rmin;
    crmax(end+1,1) = rmax;
    pipe_diameter_m(end+1,1) = D;
    pipe_length_m(end+1,1) = L;
    pipe_friction(end+1,1) = lambda;
    varargout = {arc_ids, arc_original_ids, arc_types, arc_subtypes, from, to, ...
        raw_qmin, raw_qmax, qmin, qmax, res_factor, crmin, crmax, ...
        pipe_diameter_m, pipe_length_m, pipe_friction};
end
