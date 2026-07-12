%% GasModels-compatible fmincon + SDP run for one hardcoded MatGas case
% Keep this script beside:
%   parse_matgas_instance.m
%   parse_matgas_gasmodels_ls.m
%   run_matgas_gasmodels_ls.m
%   run_matgas_gasmodels_sdp.m
%
% The SDP step requires SPOT/msspoly, CSTSS_mex, MOSEK, and the CSTSS
% recovery utilities.  If they are absent, fmincon still runs and the SDP
% result receives a descriptive skipped status.

clear;
clc;
close all;

here = fileparts(mfilename('fullpath'));
addpath(here);

% Match the path setup used by the supplied run_matgas_case when a sibling
% pathinfo/my_path setup is available, without destroying the current path.
pathinfo_dir = fullfile(here, '..', 'pathinfo');
if exist(pathinfo_dir, 'dir') == 7
    addpath(pathinfo_dir);
    if exist('my_path', 'file') == 2
        my_path;
    end
    addpath(here);
end

% -------------------------------------------------------------------------
% Select the case here.  The search checks data/ first, as requested.
% An absolute path can instead be assigned directly to matgas_file.
% -------------------------------------------------------------------------
case_name = 'gaslib-40-E-ls.m';
matgas_file = '';

candidates = { ...
    fullfile(here, 'data', case_name); ...
    fullfile(here, 'data', 'matgas', case_name); ...
    fullfile(here, '..', 'data', case_name); ...
    fullfile(here, '..', 'data', 'matgas', case_name); ...
    fullfile(pwd, 'data', case_name); ...
    fullfile(pwd, 'data', 'matgas', case_name); ...
    fullfile(here, case_name); ...
    fullfile(pwd, case_name)};

if isempty(matgas_file)
    path_match = which(case_name);
    if ~isempty(path_match)
        candidates{end+1, 1} = path_match;
    end
    for k = 1:numel(candidates)
        if exist(candidates{k}, 'file') == 2
            matgas_file = candidates{k};
            break;
        end
    end
    if isempty(matgas_file)
        attempted = sprintf('  %s\n', candidates{:});
        error('run_matgas_gasmodels_case:MissingCase', ...
            ['Could not find %s. Locations checked:\n%s', ...
             'Edit case_name or matgas_file near the top of this script.'], ...
            case_name, attempted);
    end
elseif exist(matgas_file, 'file') ~= 2
    error('run_matgas_gasmodels_case:MissingExplicitCase', ...
        'The explicitly assigned MatGas file does not exist: %s', matgas_file);
end

fprintf('Using MatGas case: %s\n', matgas_file);

% -------------------------------------------------------------------------
% fmincon settings -- retain SQP to avoid the singular barrier system from
% the earlier qabs lifting.
% -------------------------------------------------------------------------
opts = struct();
opts.verbose = true;
opts.num_starts = 8;
opts.random_seed = 7;
opts.algorithm = 'sqp';
opts.start_flow_fraction = 0.01;
opts.feasibility_tolerance = 1e-6;
opts.fmincon_display = 'off';

% -------------------------------------------------------------------------
% Moment/SOS relaxation settings, analogous to the supplied driver.
% -------------------------------------------------------------------------
opts.run_sdp = true;
opts.kappa = 2;
opts.cs_mode = 'MF';
opts.pop_ball_constraints = true;
opts.sdp_strengthen = true;
opts.report_pop_coeff_ranges = true;
opts.pop_feas_tol = 1e-6;
opts.sdp_fail_on_error = false;

[~, case_base] = fileparts(matgas_file);
opts.save_file = fullfile(here, ...
    [case_base, '-gasmodels-fmincon-sdp.mat']);

result = run_matgas_gasmodels_ls(matgas_file, opts);

fprintf('\n=== Driver summary ===\n');
fprintf('Dispatchable gas shed, fmincon : %.9g kg/s\n', ...
    result.summary.dispatchable_delivery_shed_kgps);
fprintf('Dispatchable gas served        : %.9g kg/s\n', ...
    result.summary.dispatchable_delivery_served_kgps);
if ~isempty(fieldnames(result.gasmodels_benchmark))
    fprintf('GasModels WP served reference  : %.9g kg/s\n', ...
        result.gasmodels_benchmark.dispatchable_served_kgps);
end
fprintf('SDP status                     : %s\n', result.sdp.status);

if strcmp(result.sdp.status, 'solved')
    fprintf('SDP shed lower bound            : %.9g\n', ...
        result.sdp.lower_bound);
    fprintf('fmincon minus SDP shed gap      : %.9g\n', ...
        result.sdp.absolute_gap);
    fprintf('SDP served upper bound          : %.9g\n', ...
        result.sdp.weighted_served_upper_bound);
end
fprintf('Saved result                    : %s\n', opts.save_file);
