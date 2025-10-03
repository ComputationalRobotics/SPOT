% MGSP toy instance (4-node) solved with fmincon (NLP) and SDP Relaxations
% Variables x = [p0 p1 p2 p3 q01 q12 q13 g0 s2 s3 q01_abs q12_abs q13_abs]'.
% Objective: minimize w2*s2 + w3*s3
% Constraints:
%   Linear equalities (mass balance)
%   Nonlinear equalities (Weymouth on each pipe, bidirectional)
%   Abs-value link: q_abs^2 = q^2 and q_abs >= 0
%   Bounds on pressures, supply, shedding

% clean and addpath
clc; clear; close all;
restoredefaultpath;
addpath("../pathinfo/");
my_path;

%% ------------------------ Data ------------------------
% Demands
d2 = 2;  % at node 2
d3 = 2;  % at node 3

% Pressure bounds (bar) for all nodes
pmin = 0.5; pmax = 2;

% Supply cap at node 0
g0_max = 2;

% Weymouth coefficients K_ij (unit-consistent with flows & pressures)
K01 = 1;
K12 = 1;
K13 = 1;

% Shedding weights
w2 = 1;
w3 = 3;

%% ------------------------ Variable indexing ------------------------
% x = [ p0   p1   p2   p3   q01  q12  q13  g0   s2   s3   q01_abs q12_abs q13_abs ]
i_p0 = 1;  i_p1 = 2;  i_p2 = 3;  i_p3 = 4;
i_q01 = 5; i_q12 = 6; i_q13 = 7;
i_g0 = 8;  i_s2 = 9;  i_s3 = 10;
i_q01_abs = 11; i_q12_abs = 12; i_q13_abs = 13;

nvar = 13;

%% ------------------------ Objective ------------------------
obj = @(x) w2*x(i_s2) + w3*x(i_s3);

%% ------------------------ Linear equalities (mass balance) ---------
% Node 0: g0 - q01 = 0
% Node 1: q01 - q12 - q13 = 0
% Node 2: q12 + s2 = d2
% Node 3: q13 + s3 = d3

Aeq = zeros(4, nvar);
beq = zeros(4, 1);

% Node 0
Aeq(1, i_g0)  = 1;
Aeq(1, i_q01) = -1;
beq(1) = 0;

% Node 1
Aeq(2, i_q01) = 1;
Aeq(2, i_q12) = -1;
Aeq(2, i_q13) = -1;
beq(2) = 0;

% Node 2
Aeq(3, i_q12) = 1;
Aeq(3, i_s2)  = 1;
beq(3) = d2;

% Node 3
Aeq(4, i_q13) = 1;
Aeq(4, i_s3)  = 1;
beq(4) = d3;

%% ------------------------ Bounds -----------------------------------
lb = -inf(nvar,1); ub = inf(nvar,1);

% Pressures
lb([i_p0 i_p1 i_p2 i_p3]) = pmin;
ub([i_p0 i_p1 i_p2 i_p3]) = pmax;

% Flows are now bidirectional (signed); no nonnegativity bounds.

% Supply bounds
lb(i_g0) = 0; ub(i_g0) = g0_max;

% Shedding nonnegative
lb([i_s2 i_s3]) = 0;

% |q| variables: we will enforce q_abs >= 0 in the nonlinear constraints,
% so no bound is strictly necessary. (You could also set lb to 0.)

%% ------------------------ Nonlinear constraints --------------------
% Bidirectional Weymouth:
% (p0^2 - p1^2) = K01 * q01 * q01_abs, etc.
% Abs-value link:
% q01_abs^2 = q01^2, q12_abs^2 = q12^2, q13_abs^2 = q13^2
% and q01_abs >= 0, q12_abs >= 0, q13_abs >= 0
nonlcon = @(x) weymouth_bidirectional( ...
    x, i_p0,i_p1,i_p2,i_p3, ...
    i_q01,i_q12,i_q13, ...
    i_q01_abs,i_q12_abs,i_q13_abs, ...
    K01,K12,K13);

%% -------------------- Initial guess for nlp ------------------------
x0 = randn(nvar,1);
% make a reasonable start for pressures, supplies, shedding, and q_abs
% x0([i_p0 i_p1 i_p2 i_p3]) = pmin + (pmax-pmin)*rand(4,1);
% x0(i_g0)  = 0.5*g0_max;
% x0([i_s2 i_s3]) = rand(2,1);
% x0([i_q01_abs i_q12_abs i_q13_abs]) = abs(randn(3,1)); % start nonnegative

%% ------------------------ Solve with nlp ---------------------------
opts = optimoptions('fmincon', ...
    'Algorithm','interior-point', ...
    'Display','iter', ...
    'SpecifyObjectiveGradient', false, ...
    'SpecifyConstraintGradient', false, ...
    'MaxIterations', 1000, ...
    'OptimalityTolerance', 1e-8, ...
    'StepTolerance', 1e-10, ...
    'ConstraintTolerance', 1e-8);

[x,fval,exitflag,output] = fmincon(obj, x0, [],[], Aeq,beq, lb,ub, nonlcon, opts);

%% ------------------------ Report -----------------------------------
sol.x = x;
sol.fval = fval;
sol.exitflag = exitflag;
sol.output = output;

fprintf('\n=== MGSP Solution ===\n');
fprintf('Objective (weighted shedding): %.6f\n', fval);
fprintf('Pressures [bar]: p0=%.4f, p1=%.4f, p2=%.4f, p3=%.4f\n', x(i_p0), x(i_p1), x(i_p2), x(i_p3));
fprintf('Flows      : q01=%.6f, q12=%.6f, q13=%.6f\n', x(i_q01), x(i_q12), x(i_q13));
fprintf('Flow |.|   : |q01|=%.6f, |q12|=%.6f, |q13|=%.6f\n', x(i_q01_abs), x(i_q12_abs), x(i_q13_abs));
fprintf('Supply     : g0 = %.6f\n', x(i_g0));
fprintf('Shedding   : s2 = %.6f, s3 = %.6f\n', x(i_s2), x(i_s3));

% Check residuals
[c,ceq] = nonlcon(x);
lin_resid = norm(Aeq*x - beq, inf);
fprintf('Max linear balance residual: %.3e\n', lin_resid);
fprintf('Max Weymouth/abs residual  : %.3e\n', norm(ceq, inf));
fprintf('Max inequality violation   : %.3e (<=0 is feasible)\n', max(c));
fprintf('Exitflag: %d  (see sol.output for details)\n', exitflag);

%% ------------------------ Solve using SDP Relaxations --------------
% define polynomial variables
z = msspoly('z', nvar);
% define objective function
objective = w2*z(i_s2) + w3*z(i_s3);
% define equality constraints
equality = Aeq * z - beq;
equality = [equality;
            z(i_p0)^2 - z(i_p1)^2 - K01*z(i_q01)*z(i_q01_abs);
            z(i_p1)^2 - z(i_p2)^2 - K12*z(i_q12)*z(i_q12_abs);
            z(i_p1)^2 - z(i_p3)^2 - K13*z(i_q13)*z(i_q13_abs);
            z(i_q01)^2 - z(i_q01_abs)^2;
            z(i_q12)^2 - z(i_q12_abs)^2;
            z(i_q13)^2 - z(i_q13_abs)^2;
            ];
% define inequality constraints
% inequality = [100-z'*z; % add a ball constraint on all variables 
%               z([i_p0 i_p1 i_p2 i_p3]) - pmin];
inequality = [z([i_p0 i_p1 i_p2 i_p3]) - pmin];
inequality = [inequality;
              pmax - z([i_p0 i_p1 i_p2 i_p3])];
inequality = [inequality;
              z([i_q01_abs i_q12_abs i_q13_abs])];
inequality = [inequality;
              z(i_g0);
              g0_max - z(i_g0);
              z([i_s2 i_s3])];

% call SPOT to formulate Moment-SOS SDP relaxations
if_mex = true; params.if_mex = if_mex;
kappa = 2; params.kappa = kappa; % relaxation order
relax_mode = "MOMENT"; params.relax_mode = relax_mode;
cs_mode = "MF"; params.cs_mode = cs_mode;
ts_mode = "NON"; params.ts_mode = ts_mode;
ts_mom_mode = "NON"; params.ts_mom_mode = ts_mom_mode;
ts_eq_mode = "NON"; params.ts_eq_mode = ts_eq_mode; 
if_solve = true; params.if_solve = if_solve;
params.cliques = [];
[result, res, coeff_info, aux_info] = CSTSS_mex(objective, ...
    inequality, equality, kappa, z, params);

% extract solution
blk = cell(size(aux_info.clique_size, 1), 2);
for i = 1: size(aux_info.clique_size, 1)
    blk{i, 1} = 's';
    blk{i, 2} = aux_info.clique_size(i);
end
[Xopt, yopt, Sopt, obj] = recover_mosek_sol_blk(res, blk);
if relax_mode == "MOMENT"
    Xs = Xopt;
else
    Xs = Sopt;
    for i = 1: length(Xs)
        Xs{i} = -Xs{i};
    end
end
eta = abs(fval - obj(1))/(1 + abs(fval) + abs(obj(1)));


fprintf('Upper bound on global optimum (from NLP): %.3e\n', fval);
fprintf('Lower bound on global optimum (from SDP): %.3e\n', obj(1));
fprintf('Certified relative suboptimality: %.3e\n', eta);



%% ------------------------ Helper: Nonlinear constraints -------------
function [c, ceq] = weymouth_bidirectional( ...
    x, i_p0,i_p1,i_p2,i_p3, ...
    i_q01,i_q12,i_q13, ...
    i_q01_abs,i_q12_abs,i_q13_abs, ...
    K01,K12,K13)

p0 = x(i_p0); p1 = x(i_p1); p2 = x(i_p2); p3 = x(i_p3);
q01 = x(i_q01); q12 = x(i_q12); q13 = x(i_q13);
q01_abs = x(i_q01_abs); q12_abs = x(i_q12_abs); q13_abs = x(i_q13_abs);

% Inequalities: enforce q_abs >= 0  ->  -q_abs <= 0
c = zeros(3,1);
c(1) = -q01_abs;
c(2) = -q12_abs;
c(3) = -q13_abs;

% Equalities:
% Bidirectional Weymouth using q*abs(q)
% Plus the abs-value link q_abs^2 = q^2
ceq = zeros(6,1);
ceq(1) = (p0^2 - p1^2) - K01*(q01 * q01_abs);
ceq(2) = (p1^2 - p2^2) - K12*(q12 * q12_abs);
ceq(3) = (p1^2 - p3^2) - K13*(q13 * q13_abs);

ceq(4) = q01_abs^2 - q01^2;
ceq(5) = q12_abs^2 - q12^2;
ceq(6) = q13_abs^2 - q13^2;
end
