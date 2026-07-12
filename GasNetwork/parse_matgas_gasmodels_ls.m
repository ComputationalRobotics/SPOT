function data = parse_matgas_gasmodels_ls(matgas_file, opts)
%PARSE_MATGAS_GASMODELS_LS GasModels-compatible MatGas load-shed parser.
%
%   data = parse_matgas_gasmodels_ls(matgas_file)
%   data = parse_matgas_gasmodels_ls(matgas_file, opts)
%
% This is a strict wrapper around parse_matgas_instance.  It keeps each
% dispatchable receipt, delivery, and transfer as a separate boundary and
% constructs the nodal constants used by GasModels' build_ls formulation.
% It also reads compressor directionality (column 15), preserves reverse
% compressor flow, and applies GasModels' network-wide flow cap.
%
% The existing parse_matgas_instance.m is not modified.  This file must be
% on the MATLAB path beside it.

    if nargin < 1 || isempty(matgas_file)
        error('parse_matgas_gasmodels_ls:MissingFile', ...
            'A MatGas .m file is required.');
    end
    if nargin < 2 || isempty(opts)
        opts = struct();
    end

    opts = local_defaults(opts);

    % These choices reproduce GasModels semantics.  They intentionally
    % override similarly named options supplied by the caller.
    popts = opts;
    popts.passive_bidirectional = false;
    popts.enforce_compressor_forward = false;
    popts.weight_default = 1;
    popts.weights = struct();
    popts.matgas_demand_target = 'auto';
    popts.matgas_supply_target = 'auto';
    popts.matgas_use_priorities = true;
    popts.matgas_fix_slack_pressure = true;
    popts.matgas_cap_by_network_flow = false;
    popts.matgas_map_valves = false;
    popts.matgas_map_regulators = false;
    popts.matgas_transfer_mode = 'nominal';
    popts.matgas_error_on_unsupported = true;
    popts.matgas_error_on_unknown_extensions = true;
    popts.verbose = false;

    data = parse_matgas_instance(matgas_file, popts);
    N = data.N;

    r = data.matgas.receipt;
    d = data.matgas.delivery;
    t = data.matgas.transfer;

    rdisp = logical(r.is_dispatchable);
    ddisp = logical(d.is_dispatchable);
    tdisp = logical(t.is_dispatchable);

    gm = struct();
    gm.dispatchable_receipt = local_subset_boundary(r, rdisp);
    gm.dispatchable_delivery = local_subset_boundary(d, ddisp);
    gm.dispatchable_transfer = local_subset_boundary(t, tdisp);
    gm.fixed_receipt = local_subset_boundary(r, ~rdisp);
    gm.fixed_delivery = local_subset_boundary(d, ~ddisp);
    gm.fixed_transfer = local_subset_boundary(t, ~tdisp);

    gm.fixed_supply_by_node = local_accum(N, ...
        r.node_index(~rdisp), r.nominal(~rdisp));
    gm.fixed_demand_by_node = local_accum(N, ...
        d.node_index(~ddisp), d.nominal(~ddisp));
    gm.fixed_transfer_by_node = local_accum(N, ...
        t.node_index(~tdisp), t.nominal(~tdisp));
    gm.dispatchable_delivery_max_by_node = local_accum(N, ...
        d.node_index(ddisp), d.maximum(ddisp));

    % A positive transfer is a withdrawal and a negative transfer is an
    % injection.  With out-minus-in incidence, dispatchable variables obey
    %     B*q - receipt - shed + transfer = balance_rhs.
    gm.fixed_net_injection_by_node = gm.fixed_supply_by_node ...
        - gm.fixed_demand_by_node - gm.fixed_transfer_by_node;
    gm.balance_rhs = gm.fixed_net_injection_by_node ...
        - gm.dispatchable_delivery_max_by_node;

    gm.fixed_supply_total = sum(r.nominal(~rdisp));
    gm.fixed_demand_total = sum(d.nominal(~ddisp));
    gm.fixed_transfer_total = sum(t.nominal(~tdisp));
    gm.dispatchable_supply_min = sum(r.minimum(rdisp));
    gm.dispatchable_supply_max = sum(r.maximum(rdisp));
    gm.dispatchable_demand_min = sum(d.minimum(ddisp));
    gm.dispatchable_demand_max = sum(d.maximum(ddisp));
    gm.dispatchable_transfer_min = sum(t.minimum(tdisp));
    gm.dispatchable_transfer_max = sum(t.maximum(tdisp));

    if any(gm.dispatchable_delivery.maximum ...
            < gm.dispatchable_delivery.minimum - opts.bound_tolerance)
        error('parse_matgas_gasmodels_ls:DeliveryBounds', ...
            'A dispatchable delivery has maximum below minimum.');
    end
    gm.dispatchable_delivery.shed_max = ...
        gm.dispatchable_delivery.maximum ...
        - gm.dispatchable_delivery.minimum;

    % GasModels correct_f_bounds! clips every component flow to the maximum
    % possible network injection: positive receipt maxima plus the magnitude
    % of any negative transfer minima.  (Storage is rejected by the base
    % parser.)  This is 2162 kg/s for gaslib-40-E-ls.m.
    gm.network_flow_cap = sum(max(r.maximum, 0)) ...
        - sum(min(t.minimum, 0));
    if ~(isfinite(gm.network_flow_cap) && gm.network_flow_cap > 0)
        error('parse_matgas_gasmodels_ls:FlowCap', ...
            'The sum of active receipt maxima must be positive and finite.');
    end
    data.qmin = max(data.qmin, -gm.network_flow_cap);
    data.qmax = min(data.qmax,  gm.network_flow_cap);

    % parse_matgas_instance deliberately exposes original arc ids, so the
    % compressor table can be matched without relying on row order.
    meta = local_read_compressor_metadata(matgas_file);
    directionality = nan(data.M, 1);
    power_max_raw = nan(data.M, 1);
    compressor_arcs = find(strcmp(data.matgas.arc_subtypes, 'compressor'));
    active_meta = find(meta.status ~= 0);
    if numel(compressor_arcs) ~= numel(active_meta)
        error('parse_matgas_gasmodels_ls:CompressorCount', ...
            ['The active compressor count in the literal table does not ', ...
             'match the parsed network.']);
    end

    for k = 1:numel(compressor_arcs)
        e = compressor_arcs(k);
        original_id = char(data.matgas.arc_original_ids{e});
        hit = find(strcmp(meta.id, original_id) & meta.status ~= 0);
        if numel(hit) ~= 1
            error('parse_matgas_gasmodels_ls:CompressorId', ...
                'Could not uniquely match active compressor id %s.', original_id);
        end
        directionality(e) = meta.directionality(hit);
        power_max_raw(e) = meta.power_max(hit);

        if ~ismember(directionality(e), [0, 1, 2])
            error('parse_matgas_gasmodels_ls:Directionality', ...
                'Unsupported directionality %.9g on compressor %s.', ...
                directionality(e), original_id);
        end
        if directionality(e) == 1
            data.qmin(e) = max(data.qmin(e), 0);
        end
        if data.qmin(e) > data.qmax(e) + opts.bound_tolerance
            error('parse_matgas_gasmodels_ls:CompressorFlowBounds', ...
                'Directionality conflicts with flow bounds on compressor %s.', ...
                original_id);
        end
        if power_max_raw(e) < opts.unbounded_power_threshold
            error('parse_matgas_gasmodels_ls:FiniteCompressorPower', ...
                ['Compressor %s has a finite power limit.  The separate ', ...
                 'MATLAB routine currently targets the GasLib-40 benchmark ', ...
                 'whose compressor power limits are nonbinding (1e100).'], ...
                original_id);
        end
    end

    if any(data.qmin > data.qmax + opts.bound_tolerance)
        error('parse_matgas_gasmodels_ls:ArcFlowBounds', ...
            'GasModels flow correction produced an empty arc flow interval.');
    end

    gm.compressor_directionality = directionality;
    gm.compressor_power_max_raw = power_max_raw;
    gm.compressor_arc_index = compressor_arcs;
    gm.formulation = 'GasModels load shedding / squared-pressure WP';
    gm.parser_options_forced = struct( ...
        'passive_bidirectional', false, ...
        'enforce_compressor_forward', false, ...
        'matgas_fix_slack_pressure', true, ...
        'matgas_use_priorities', true);
    data.gasmodels_ls = gm;

    if opts.verbose
        fprintf('\n=== GasModels-compatible MatGas parse ===\n');
        fprintf('File                         : %s\n', char(matgas_file));
        fprintf('Fixed receipt / delivery     : %.9g / %.9g kg/s\n', ...
            gm.fixed_supply_total, gm.fixed_demand_total);
        fprintf('Dispatch receipt max         : %.9g kg/s\n', ...
            gm.dispatchable_supply_max);
        fprintf('Dispatch delivery max        : %.9g kg/s\n', ...
            gm.dispatchable_demand_max);
        fprintf('Dispatch boundaries (R/D/T)  : %d / %d / %d\n', ...
            numel(gm.dispatchable_receipt.id), ...
            numel(gm.dispatchable_delivery.id), ...
            numel(gm.dispatchable_transfer.id));
        fprintf('GasModels network flow cap   : %.9g kg/s\n', ...
            gm.network_flow_cap);
        if ~isempty(compressor_arcs)
            fprintf('Compressor directionality    : %s\n', ...
                mat2str(directionality(compressor_arcs).'));
        end
    end
end

function opts = local_defaults(opts)
    opts = local_set_default(opts, 'verbose', true);
    opts = local_set_default(opts, 'bound_tolerance', 1e-9);
    opts = local_set_default(opts, 'unbounded_power_threshold', 1e90);
end

function s = local_set_default(s, name, value)
    if ~isfield(s, name) || isempty(s.(name))
        s.(name) = value;
    end
end

function out = local_subset_boundary(in, mask)
    mask = logical(mask(:));
    out = struct();
    names = fieldnames(in);
    for k = 1:numel(names)
        name = names{k};
        value = in.(name);
        if isempty(value)
            out.(name) = value;
        elseif size(value, 1) == numel(mask)
            out.(name) = value(mask, :);
        else
            error('parse_matgas_gasmodels_ls:BoundaryShape', ...
                'Unexpected shape for boundary field %s.', name);
        end
    end
end

function values = local_accum(N, nodes, amounts)
    if isempty(nodes)
        values = zeros(N, 1);
    else
        values = accumarray(nodes(:), amounts(:), [N, 1], @sum, 0);
    end
end

function meta = local_read_compressor_metadata(matgas_file)
    text = fileread(matgas_file);
    text = local_strip_matlab_comments(text);
    token = regexp(text, ...
        '(?s)\<mgc\s*\.\s*compressor\s*=\s*\[(.*?)\]\s*;', ...
        'tokens', 'once');

    meta = struct();
    meta.id = cell(0, 1);
    meta.power_max = zeros(0, 1);
    meta.status = zeros(0, 1);
    meta.directionality = zeros(0, 1);
    if isempty(token)
        return;
    end

    rows = regexp(token{1}, ';|\r\n|\n|\r', 'split');
    for row_number = 1:numel(rows)
        row = strtrim(rows{row_number});
        if isempty(row)
            continue;
        end
        fields = regexp(row, '''(?:''''|[^''])*''|[^\s,;]+', 'match');
        if numel(fields) < 15
            error('parse_matgas_gasmodels_ls:CompressorColumns', ...
                ['Compressor table row %d has fewer than 15 columns; ', ...
                 'directionality cannot be recovered exactly.'], row_number);
        end
        id = local_clean_id(fields{1});
        power = local_number(fields{6}, 'power_max', row_number);
        status = local_number(fields{13}, 'status', row_number);
        direction = local_number(fields{15}, 'directionality', row_number);
        if abs(status - round(status)) > 1e-10 || ...
                abs(direction - round(direction)) > 1e-10
            error('parse_matgas_gasmodels_ls:CompressorInteger', ...
                'Status and directionality must be integers on row %d.', ...
                row_number);
        end
        meta.id{end+1, 1} = id; %#ok<AGROW>
        meta.power_max(end+1, 1) = power; %#ok<AGROW>
        meta.status(end+1, 1) = round(status); %#ok<AGROW>
        meta.directionality(end+1, 1) = round(direction); %#ok<AGROW>
    end

    if numel(unique(meta.id)) ~= numel(meta.id)
        error('parse_matgas_gasmodels_ls:DuplicateCompressor', ...
            'Duplicate compressor ids were found.');
    end
end

function clean = local_strip_matlab_comments(text)
    % Preserve newlines and quoted percent signs while removing both line
    % comments and MATLAB %{ ... %} block comments.
    clean = text;
    in_string = false;
    in_block = false;
    i = 1;
    n = numel(text);
    while i <= n
        ch = text(i);
        if in_block
            if ch == '%' && i < n && text(i + 1) == '}'
                clean(i:i+1) = '  ';
                in_block = false;
                i = i + 2;
            else
                if ch ~= char(10) && ch ~= char(13)
                    clean(i) = ' ';
                end
                i = i + 1;
            end
        elseif in_string
            if ch == ''''
                if i < n && text(i + 1) == ''''
                    i = i + 2;
                else
                    in_string = false;
                    i = i + 1;
                end
            else
                i = i + 1;
            end
        elseif ch == ''''
            in_string = true;
            i = i + 1;
        elseif ch == '%'
            if i < n && text(i + 1) == '{'
                clean(i:i+1) = '  ';
                in_block = true;
                i = i + 2;
            else
                while i <= n && text(i) ~= char(10) && text(i) ~= char(13)
                    clean(i) = ' ';
                    i = i + 1;
                end
            end
        else
            i = i + 1;
        end
    end
end

function id = local_clean_id(token)
    token = strtrim(token);
    if numel(token) >= 2 && token(1) == '''' && token(end) == ''''
        id = strrep(token(2:end-1), '''''', '''');
    else
        id = token;
    end
end

function value = local_number(token, field, row)
    value = str2double(token);
    if ~isfinite(value)
        error('parse_matgas_gasmodels_ls:NumericLiteral', ...
            'Invalid %s value on compressor row %d.', field, row);
    end
end
