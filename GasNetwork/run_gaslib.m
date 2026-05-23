%RUN_GASLIB  Run GasLib test instances through run_gaslib_mgsp (NLP, then SDP).
%
% Three instances under GasNetwork/data/ (set instance_key below):
%   gaslib40_original  GasLib-40-v1-20211130.net + .scn, scn_id = nomination_1
%   gaslib40_modified  GasLib-40-v1-20211130.net + GasLib-40-gasmodels-ls-converted.scn
%   gaslib135          GasLib-135-v1-20211130.net + .scn, scn_id = nomination_1
%
% Requires SPOT (msspoly) and CSTSS_mex for the SDP step.
%
% Usage (from MATLAB, with GasNetwork on the path):
%   run_gaslib

% clean and addpath
clc; clear; close all;
restoredefaultpath;
addpath("../pathinfo/");
my_path;

thisdir = fileparts(mfilename('fullpath'));
datadir = fullfile(thisdir, 'data');

% --- instance selector: 'gaslib40_original' | 'gaslib40_modified' | 'gaslib135'
instance_key = 'gaslib135';

instances.gaslib40_original = struct( ...
    'label', 'GasLib-40 (original)', ...
    'net_file', fullfile(datadir, 'GasLib-40-v1-20211130.net'), ...
    'scn_file', fullfile(datadir, 'GasLib-40-v1-20211130.scn'), ...
    'scn_id', 'nomination_1');

instances.gaslib40_modified = struct( ...
    'label', 'GasLib-40 (gasmodels LS)', ...
    'net_file', fullfile(datadir, 'GasLib-40-v1-20211130.net'), ...
    'scn_file', fullfile(datadir, 'GasLib-40-gasmodels-ls-converted.scn'), ...
    'scn_id', 'gasmodels_ls_mgs_bounds');  % alt: 'gasmodels_ls_fixed_nominal'

instances.gaslib135 = struct( ...
    'label', 'GasLib-135', ...
    'net_file', fullfile(datadir, 'GasLib-135-v1-20211130.net'), ...
    'scn_file', fullfile(datadir, 'GasLib-135-v1-20211130.scn'), ...
    'scn_id', 'nomination_1');

if ~isfield(instances, instance_key)
    error('run_gaslib40:Instance', ...
        'Unknown instance_key ''%s''. Use gaslib40_original, gaslib40_modified, or gaslib135.', ...
        instance_key);
end
inst = instances.(instance_key);
net_file = inst.net_file;
scn_file = inst.scn_file;
scn_id = inst.scn_id;
fprintf('Instance: %s\n  net: %s\n  scn: %s  (id=%s)\n', ...
    inst.label, net_file, scn_file, scn_id);

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


% plot the histogram of clique sizes, and also correlative sparsity pattern graph
params = out.sdp.relax_info;
cliques = params.cliques;
n = params.total_var_num;
C = containers.Map('KeyType', 'uint64', 'ValueType', 'any');

% Print the maximum number of variables in cliques and the number of cliques
clique_sizes = cellfun(@length, cliques);
max_clique_size = max(clique_sizes);
num_cliques = numel(cliques);
fprintf('Maximum number of variables in cliques: %d\n', max_clique_size);
fprintf('Number of cliques: %d\n', num_cliques);

% Plot the histogram of clique sizes
figure;
histogram(clique_sizes);
xlabel('Clique size', 'FontSize', 30);
ylabel('Frequency', 'FontSize', 30);
set(gca, 'FontSize', 24);


%% embed graph
figure;
% suppose we only have CS
for clique_id = 1: length(cliques)
    clique = cliques{clique_id};
    clique_size = length(clique);

    for i = 1: clique_size
        for j = i+1: clique_size
            src = clique(i);
            dst = clique(j);
            key = src * (n+1) + dst;
            C(key) = [src, dst];
        end
    end

    % for i = 1: clique_size
    %     src = clique(i);
    %     if i == clique_size 
    %         dst = clique(1);
    %     else
    %         dst = clique(i+1);
    %     end
    %     key = src * (n+1) + dst;
    %     C(key) = [src, dst];
    % end
end

value_set = values(C);
edge_num = length(value_set);
edges = zeros(edge_num, 2);
for i = 1: edge_num 
    edges(i, :) = value_set{i};
end

G = graph(edges(:,1), edges(:,2));
h = plot(G, 'Layout', 'force');
x = h.XData;
y = h.YData;
close;

%% get convex hull of each clique
convex_hulls = cell(length(cliques), 1);
effective_id_num = 0;
for clique_id = 1: length(cliques)
    clique = cliques{clique_id};
    x_clique = x(clique);
    y_clique = y(clique);
    if length(clique) > 2
        effective_id_num = effective_id_num + 1;
        k = convhull(x_clique', y_clique');
        x_convexhull = x_clique(k);
        y_convexhull = y_clique(k);
    else
        x_convexhull = x_clique;
        y_convexhull = y_clique;
    end
    convex_hulls{clique_id} = [x_convexhull; y_convexhull];
end

%% get approximated minimal enclosing ellipse for each clique
ellipses = zeros(effective_id_num, 5);
for clique_id = 1: effective_id_num
    convexhull = convex_hulls{clique_id};
    xh = convexhull(1, 1:end-1);
    yh = convexhull(2, 1:end-1);
    [xc, yc, a, b, theta] = get_approx_min_enclosing_ellipse(xh, yh);
    ellipses(clique_id, :) = [xc, yc, a, b, theta];
end



%% draw cleaned embedding
% figure;
% scatter(x, y, 10, 'b');
% hold on;
% for clique_id = 1: length(cliques)
%     convexhull = convex_hulls{clique_id};
%     plot(convexhull(1, :), convexhull(2, :), 'Color', [0, 0, 0], 'LineWidth', 0.2);
% end

%% draw final embedding 
figure('Position', [100, 100, 800, 800]);
G = graph(edges(:,1), edges(:,2));
h = plot(G, 'Layout', 'force');

% Customize node properties
h.NodeColor = 'blue';            % Change all node colors to red
h.MarkerSize = 5.0;              % Change node size
h.Marker = 'o';                 % Node shape (default is circle)
h.NodeLabel = {};               % Remove node labels for a clean look

% Customize edge properties
h.LineWidth = 0.4;                % Change edge width
h.EdgeColor = [0, 0, 1];        % Change edge color to blue
h.EdgeAlpha = 0.35;              % Set edge transparency

hold on;
for clique_id = 1: effective_id_num
    data = ellipses(clique_id, :);
    xc = data(1); yc = data(2); a = data(3); b = data(4); theta = data(5);
    t = linspace(0, 2*pi, 100);
    x_ellipse = xc + a * cos(t) * cos(theta) - b * sin(t) * sin(theta);
    y_ellipse = yc + a * cos(t) * sin(theta) + b * sin(t) * cos(theta);
    plot(x_ellipse, y_ellipse, 'r', 'LineWidth', 2.0);
end

exportgraphics(gcf, 'test.tiff', 'Resolution', 600);
print('-dpng', '-r300', "test.png");

%% helper functions
function [xc, yc, a, b, theta] = get_approx_min_enclosing_ellipse(xh, yh)
    % Objective function: minimize ellipse area pi * a * b
    areaFunc = @(p) pi * p(3) * p(4); % p = [xc, yc, a, b, theta]
    
    % Constraints: all points must satisfy the ellipse equation
    ellipseConstraint = @(p) arrayfun(@(i) ...
        ( cos(p(5)) * (xh(i) - p(1)) + sin(p(5)) * (yh(i) - p(2)) )^2 / p(3)^2 + ...
        ( sin(p(5)) * (xh(i) - p(1)) - cos(p(5)) * (yh(i) - p(2)) )^2 / p(4)^2 - 1, 1:length(xh));
    
    % Initial guess for [xc, yc, a, b, theta]
    p0 = [mean(xh), mean(yh), range(xh), range(yh), 0];
    
    % Bounds for optimization: a, b > 0
    lb = [-inf, -inf, 0, 0, -pi];
    ub = [inf, inf, inf, inf, pi];
    
    % Solve the optimization problem
    opts = optimoptions('fmincon', 'Display', 'final', 'Algorithm', 'sqp');
    p = fmincon(areaFunc, p0, [], [], [], [], lb, ub, ...
        @(p) deal(ellipseConstraint(p), []), opts);
    
    % Extract the ellipse parameters
    xc = p(1); yc = p(2); % Center
    a = p(3); b = p(4);   % Semi-axes
    theta = p(5);         % Rotation angle
end
