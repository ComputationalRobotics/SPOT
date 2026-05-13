%RUN_GASLIB40  Run GasLib-40 through run_gaslib_mgsp (NLP, then SDP relaxation).
%
% Expects GasLib-40 files under GasNetwork/data/:
%   GasLib-40-v1-20211130.net
%   GasLib-40-v1-20211130.scn
%
% Requires SPOT (msspoly) and CSTSS_mex for the SDP step.
%
% Usage (from MATLAB, with GasNetwork on the path):
%   run_gaslib40

% clean and addpath
clc; clear; close all;
restoredefaultpath;
addpath("../pathinfo/");
my_path;

thisdir = fileparts(mfilename('fullpath'));
net_file = fullfile(thisdir, 'data', 'GasLib-40-v1-20211130.net');
% scn_file = fullfile(thisdir, 'data', 'GasLib-40-v1-20211130.scn');
scn_file = fullfile(thisdir, 'data', 'GasLib-40-gasmodels-ls-converted.scn');
scn_id = 'gasmodels_ls_mgs_bounds';
% scn_id = 'gasmodels_ls_fixed_nominal';

opts = struct();
opts.run_sdp = true;
opts.kappa = 2;  % moment relaxation order (passed to CSTSS_mex)
opts.cs_mode = 'MF';
opts.report_pop_coeff_ranges = true;
opts.pop_ball_constraints = true;
opts.pop_ball_c_inf = 200;   % optional
opts.verify_nlp_pop_hat = true;

% Optional: override fmincon limits (otherwise run_gaslib_mgsp defaults apply,
% including MaxFunctionEvaluations = 50000 when opts.fmincon_options is empty).
opts.fmincon_options = optimoptions('fmincon', ...
    'Algorithm', 'interior-point', ...
    'EnableFeasibilityMode', true, ...
    'Display', 'iter', ...
    'SpecifyObjectiveGradient', false, ...
    'SpecifyConstraintGradient', false, ...
    'MaxIterations', 2000, ...
    'MaxFunctionEvaluations', 1e6, ...
    'OptimalityTolerance', 1e-8, ...
    'StepTolerance', 1e-10, ...
    'ConstraintTolerance', 1e-8);

out = run_gaslib_mgsp(net_file, scn_file, scn_id, opts);
