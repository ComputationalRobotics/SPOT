close all; clear; 

%% get data
prefix = "2025-01-22_00-59-22";
prefix = "/PushBoxTunnel2_MATLAB/" + prefix + "/";
data_prefix = "./data/" + prefix;
fig_prefix = "./figs/" + prefix;

filename = "data_ordered_YULIN";
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
end

load(data_prefix + "params.mat");
filename = fig_prefix + img_name;

% params
N = params.N;
dt = params.dt;
m = params.m;
g = params.g;
mu1 = params.mu1;
mu2 = params.mu2; % temporarily unused
c = params.c;
a = params.a;
b = params.b;
r = params.r;
eta = params.eta; % make sure different modes have no overlaps
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
xlim([-s_max, s_max]);  
ylim([-s_max, s_max]);

% colors
slider_color = [0, 0.5, 1];
slider_final_color = [0, 0, 0];
obs_color = [0, 0, 0];
pusher_color = 'r';

% initilize obstacles
for i = 1: size(params.obs_circles, 1)
    c = draw_circle(params.obs_circles(i, 1), params.obs_circles(i, 2), params.obs_circles(i, 3), ...
        obs_color, 1, true);
end

% initialize slider and pusher
th_final = params.th_final;
sx_final = params.sx_final;
sy_final = params.sy_final;
obj_circles = params.obj_circles;
slider_final = draw_slider(cos(th_final), sin(th_final), sx_final, sy_final, ...
    a, b, slider_final_color, 0.5, true);

% title and axis
title('Push Box Animation');
xlabel('x');
ylabel('y');

% animation
for k = 1: num_steps
    slider = draw_slider(rot_iter(1, k), rot_iter(2, k), s_iter(1, k), s_iter(2, k), ...
                        a, b, slider_color, 2.5, true);
    pusher = draw_pusher(rot_iter(1, k), rot_iter(2, k), s_iter(1, k), s_iter(2, k), ...
                        p_iter(1, k), p_iter(2, k), ...
                        a, b, pusher_color, 50, true);

    % draw obj_circles
    c_list = [];
    for ii = 1: size(obj_circles, 1)
        px = obj_circles(ii, 1);
        py = obj_circles(ii, 2);
        r = obj_circles(ii, 3);
        [wx, wy] = S2W(px, py, ...
            rot_iter(1, k), rot_iter(2, k), s_iter(1, k), s_iter(2, k));
        c = draw_circle(wx, wy, r, slider_color, 1, true);
        c_list = [c_list; c];
    end

    % draw obj_ellipse
    % e = draw_ellipse(rot_iter(1, k), rot_iter(2, k), s_iter(1, k), s_iter(2, k), ...
    %     params.a_obj, params.b_obj, slider_color, 1, true);
    
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
    for ii = 1: length(c_list)
        delete(c_list(ii));
    end
    % delete(e);
end


%% helper functions
function e = draw_ellipse(rc, rs, sx, sy, ...
    a_obj, b_obj, color, linewidth, if_hold_on)
    th = linspace(0, 2*pi, 1000);
    px = a_obj * cos(th);
    py = b_obj * sin(th);
    [wx, wy] = S2W(px, py, rc, rs, sx, sy);
    e = plot(wx, wy, 'Color', color, 'LineWidth', linewidth);
end

function c = draw_circle(sx, sy, r, color, linewidth, if_hold_on)
    th = linspace(0, 2*pi, 1000);
    x = sx + r * cos(th);
    y = sy + r * sin(th);
    c = plot(x, y, 'Color', color, 'LineWidth', linewidth);
end

function p = draw_slider(rc, rs, sx, sy, ...
    a, b, color, linewidth, if_hold_on)
    car_xa = sx + a * rc + b * rs; car_ya = sy + a * rs - b * rc;
    car_xb = sx + a * rc - b * rs; car_yb = sy + a * rs + b * rc;
    car_xc = sx - a * rc - b * rs; car_yc = sy - a * rs + b * rc;
    car_xd = sx - a * rc + b * rs; car_yd = sy - a * rs - b * rc;
    rec_vertex = [
        car_xa, car_ya;
        car_xb, car_yb;
        car_xc, car_yc;
        car_xd, car_yd;
    ];
    rec_vertex_aug = rec_vertex;
    rec_vertex_aug(end+1, :) = rec_vertex(1, :);
    p = plot(rec_vertex_aug(:, 1), rec_vertex_aug(:, 2), 'Color', color, 'LineWidth', linewidth);
    if if_hold_on
        hold on;
    end
end

function p = draw_pusher(rc, rs, sx, sy, px, py, ...
    a, b, color, s, if_hold_on)
    % slightly move pusher away from slider for visualization
    eta = 0.01;
    if abs(px^2 - a^2) < 1e-6
        px = sign(px) * (abs(px) + eta);
    elseif abs(py^2 - b^2) < 1e-6
        py = sign(py) * (abs(py) + eta);
    end
    x = sx + px * rc - py * rs;
    y = sy + px * rs + py * rc;
    p = scatter(x, y, s, color, 'filled');
    if if_hold_on
        hold on;
    end
end

function [wx, wy] = S2W(px, py, rc, rs, sx, sy)
    wx = sx + px * rc - py * rs;
    wy = sy + px * rs + py * rc; 
end

function traj = rescale_sol(v_opt, id, N, ...
    s_max, px_max, py_max, F_max)
    rot_opt = zeros(2, N+1); % (rc, rs)
    s_opt = zeros(2, N+1); % (sx, sy)
    f_opt = zeros(2, N); % (fc, fs)
    p_opt = zeros(2, N); % (px, py)
    F_opt = zeros(2, N); % (Fx, Fy)
    lam_opt = zeros(4, N); % (lam1, lam2, lam3, lam4)
    
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
    end

    traj.rot = rot_opt; 
    traj.s = s_opt;
    traj.f = f_opt;
    traj.p = p_opt;
    traj.F = F_opt;
    traj.lam = lam_opt;
end



