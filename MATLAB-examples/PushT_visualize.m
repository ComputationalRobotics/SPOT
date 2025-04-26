close all; clear; 

%% get data
prefix = "2025-01-19_08-59-06";
prefix = "/PushT_MATLAB/" + prefix + "/";
data_prefix = "./data/" + prefix;
fig_prefix = "./figs/" + prefix;

filename = "v_opt_realworld";
if filename == "data_robust_IPOPT"
    load(data_prefix + filename + ".mat");
    v = v_opt;
    img_name = "robust_IPOPT.gif";
elseif filename == "data_naive_IPOPT"
    load(data_prefix + filename + ".mat");
    v = v_opt;
    img_name = "naive_IPOPT.gif";
elseif filename == "data_ordered_IPOPT"
    load(data_prefix + filename + ".mat");
    v = v_opt;
    img_name = "ordered_IPOPT.gif";
elseif filename == "v_opt_robust"
    load(data_prefix + filename + ".mat");
    v = v_opt_robust;
    img_name = "opt_robust.gif";
elseif filename == "v_opt_naive"
    load(data_prefix + filename + ".mat");
    v = v_opt_naive;
    img_name = "opt_naive.gif";
elseif filename == "v_opt_ordered"
    load(data_prefix + filename + ".mat");
    v = v_opt_ordered;
    img_name = "opt_ordered.gif";
elseif filename == "data_ordered_YULIN";
    load(data_prefix + filename + "/data.mat");
    v = xHistory(:, end);
    img_name = "ordered_YULIN.gif";
elseif filename == "data_robust_YULIN";
    load(data_prefix + filename + "/data.mat");
    v = xHistory(:, end);
    img_name = "robust_YULIN.gif";
elseif filename == "data_naive_YULIN";
    load(data_prefix + filename + "/data.mat");
    v = xHistory(:, end);
    img_name = "naive_YULIN.gif";
elseif filename == "v_opt_realworld";
    load(data_prefix + filename + ".mat");
    v = v;
    img_name = "opt_realworld.gif";
end

load(data_prefix + "params.mat");
filename = fig_prefix + img_name;

% System parameters
N = params.N; num_steps = N;
dt = params.dt;
m = params.m;
g = params.g;
mu1 = params.mu1;
mu2 = params.mu2; % temporarily unused
c = params.c;

% Geometry information
l = params.l;
dc = params.dc; % center of mass
r = params.r;
eta = params.eta; % ensure different modes have no overlaps

% Variable maximums and minimums
s_max = params.s_max;
px_max = params.px_max;
py_max = params.py_max;
F_max = params.F_max;
fc_min = params.fc_min;
id = params.id;
                           
num_steps = N;   

frame_count = 0; 
fps = 10;
slow_rate = 5;
loop_length = max(1/fps/dt/slow_rate, 1);

% rescale variables
traj = rescale_sol(v, id, N, s_max, px_max, py_max, F_max);
rot_iter = traj.rot;
s_iter = traj.s;
p_iter = traj.p;
F_iter = traj.F;
f_iter = traj.f;

% create figure session
figure;
hold on;
axis equal;
xlim([-s_max - 2.5 * l, s_max + 2.5 * l]);  
ylim([-s_max - 2.5 * l, s_max + 2.5 * l]);

% colors
slider_color = [0, 0.5, 1];
slider_final_color = [0, 0, 0];
pusher_color = 'r';

% initialize slider and pusher
slider_final = draw_slider(cos(0), sin(0), 0, 0, ...
    l, dc, slider_final_color, 0.5, true);

% title and axis
title('Push T-block Animation');
xlabel('x');
ylabel('y');

% animation
for k = 1: num_steps
    slider = draw_slider(rot_iter(1, k), rot_iter(2, k), s_iter(1, k), s_iter(2, k), ...
                        l, dc, slider_color, 2.5, true);
    pusher = draw_pusher(rot_iter(1, k), rot_iter(2, k), s_iter(1, k), s_iter(2, k), ...
                        p_iter(1, k), p_iter(2, k), ...
                        l, dc, pusher_color, 50, true);
    
    drawnow;
    pause(dt);

    if(mod(k,loop_length)==0)
        frame = getframe(gcf);
        img = frame2im(frame);
        [img_ind, cm] = rgb2ind(img, 256);
        if frame_count == 0
            imwrite(img_ind, cm, filename, 'gif', 'LoopCount', Inf, 'DelayTime', 1/fps);
        elseif k == num_steps
            imwrite(img_ind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 2);
        else
            imwrite(img_ind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 1/fps);
        end
        frame_count = frame_count + 1;
    end

    delete(slider);
    delete(pusher);
end


function p = draw_slider(rc, rs, sx, sy, ...
    l, dc, color, linewidth, if_hold_on)
    x1 = -2.0 * l;
    x2 = -0.5 * l;
    x3 = 0.5 * l;
    x4 = 2.0 * l;
    y1 = -dc * l;
    y2 = (3.0-dc) * l;
    y3 = (4.0-dc) * l;

    rec_vertex = [
        rot(x4, y3, sx, sy, rc, rs);
        rot(x4, y2, sx, sy, rc, rs);
        rot(x3, y2, sx, sy, rc, rs);
        rot(x3, y1, sx, sy, rc, rs);
        rot(x2, y1, sx, sy, rc, rs);
        rot(x2, y2, sx, sy, rc, rs);
        rot(x1, y2, sx, sy, rc, rs);
        rot(x1, y3, sx, sy, rc, rs);
    ];
    rec_vertex_aug = rec_vertex;
    rec_vertex_aug(end+1, :) = rec_vertex(1, :);
    p = plot(rec_vertex_aug(:, 1), rec_vertex_aug(:, 2), 'Color', color, 'LineWidth', linewidth);
    if if_hold_on
        hold on;
    end
end

function p = draw_pusher(rc, rs, sx, sy, px, py, ...
    l, dc, color, s, if_hold_on)
    % slightly move pusher away from slider for visualization
    % eta = 1.3;
    % if abs(px^2 - a^2) < 1e-6
    %     px = sign(px) * abs(px) * eta;
    % elseif abs(py^2 - b^2) < 1e-6
    %     py = sign(py) * abs(py) * eta;
    % end
    x = sx + px * rc - py * rs;
    y = sy + px * rs + py * rc;
    p = scatter(x, y, s, color, 'filled');
    if if_hold_on
        hold on;
    end
end

function a = rot(x0, y0, xc, yc, rc, rs)
    x1 = xc + x0 * rc - y0 * rs;
    y1 = yc + x0 * rs + y0 * rc;
    a = [x1, y1];
end

function traj = rescale_sol(v_opt, id, N, s_max, px_max, py_max, F_max)
    % Initialize matrices
    rot_opt = zeros(2, N+1); % (rc, rs)
    s_opt = zeros(2, N+1);   % (sx, sy)
    f_opt = zeros(2, N);     % (fc, fs)
    p_opt = zeros(2, N);     % (px, py)
    F_opt = zeros(2, N);     % (Fx, Fy)
    lam_opt = zeros(8, N);   % (lam1, lam2, lam3, lam4)

    % Loop over k to populate the matrices
    for k = 0:N
        rot_opt(1, k+1) = v_opt(id("rc", k));
        rot_opt(2, k+1) = v_opt(id("rs", k));
        s_opt(1, k+1) = v_opt(id("sx", k)) * s_max;
        s_opt(2, k+1) = v_opt(id("sy", k)) * s_max;
    end

    for k = 1:N
        f_opt(1, k) = v_opt(id("fc", k));
        f_opt(2, k) = v_opt(id("fs", k));
        p_opt(1, k) = v_opt(id("px", k)) * px_max;
        p_opt(2, k) = v_opt(id("py", k)) * py_max;
        F_opt(1, k) = v_opt(id("Fx", k)) * F_max;
        F_opt(2, k) = v_opt(id("Fy", k)) * F_max;
        lam_opt(1, k) = v_opt(id("lam1", k));
        lam_opt(2, k) = v_opt(id("lam2", k));
        lam_opt(3, k) = v_opt(id("lam3", k));
        lam_opt(4, k) = v_opt(id("lam4", k));
        lam_opt(5, k) = v_opt(id("lam5", k));
        lam_opt(6, k) = v_opt(id("lam6", k));
        lam_opt(7, k) = v_opt(id("lam7", k));
        lam_opt(8, k) = v_opt(id("lam8", k));
    end

    traj.rot = rot_opt; 
    traj.s = s_opt;
    traj.f = f_opt;
    traj.p = p_opt;
    traj.F = F_opt;
    traj.lam = lam_opt;
end
