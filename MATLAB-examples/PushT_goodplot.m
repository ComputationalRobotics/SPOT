close all; clear all; 

if_draw_image = true;
if_draw_gif = true;

%% get data
prefix = "2025-01-17_08-52-15";
filename = "data_ordered_YULIN";
% choose the right name
prefix = "/PushT_MATLAB/" + prefix + "/";
data_prefix = "./data" + prefix;
fig_prefix = "./figs" + prefix;
if filename == "data_robust_IPOPT"
    load(data_prefix + filename + ".mat");
    v = v_opt;
elseif filename == "data_naive_IPOPT"
    load(data_prefix + filename + ".mat");
    v = v_opt;
elseif filename == "data_ordered_IPOPT"
    load(data_prefix + filename + ".mat");
    v = v_opt;
elseif filename == "v_opt_robust"
    load(data_prefix + filename + ".mat");
    v = v_opt_robust;
elseif filename == "v_opt_naive"
    load(data_prefix + filename + ".mat");
    v = v_opt_naive;
elseif filename == "v_opt_ordered"
    load(data_prefix + filename + ".mat");
    v = v_opt_ordered;
elseif filename == "data_ordered_YULIN";
    load(data_prefix + filename + "/data.mat");
    v = xHistory(:, end);
elseif filename == "data_robust_YULIN";
    load(data_prefix + filename + "/data.mat");
    v = xHistory(:, end);
elseif filename == "data_naive_YULIN";
    load(data_prefix + filename + "/data.mat");
    v = xHistory(:, end);
end
img_name = "goodplot_" + filename + ".png";
gif_name = "goodplot_" + filename + ".gif";
load(data_prefix + "params.mat");
img_filename = fig_prefix + img_name;
gif_filename = fig_prefix + gif_name;

%% params
N = params.N;
dt = params.dt;
m = params.m;
g = params.g;
mu1 = params.mu1;
mu2 = params.mu2; % temporarily unused
eta = params.eta; % make sure different modes have no overlaps
s_max = params.s_max;
px_max = params.px_max;
py_max = params.py_max;
F_max = params.F_max;
fc_min = params.fc_min;
id = params.id; 
vertices_box = params.vertices;
% rescale variables
traj = rescale_sol(v, id, N, s_max, px_max, py_max, F_max);
rot = traj.rot;
s = traj.s;
p = traj.p;
F = traj.F;
f = traj.f;
lam = traj.lam;
% obstacles and final goals
th_final = params.th_final;
rc_final = cos(th_final); rs_final = sin(th_final);
sx_final = params.sx_final;
sy_final = params.sy_final;

if if_draw_image
    figure; hold on;
    % final goal params
    final_plot_params.lw = 0.5; final_plot_params.color = [0.0, 0.0, 0.0]; final_plot_params.alpha = 0.5;
    % pusher params
    pusher_plot_params.s = 25; pusher_plot_params.alpha = 1.0;
    % slider params
    slider_plot_params.lw = 1.0; slider_plot_params.alpha = 1.0;
    % force params
    force_plot_params.lw = 1.0; force_plot_params.color = [1, 0, 0]; force_plot_params.arrow_size = 1.5;
    unit_length = 0.12;
    % time params
    dt_fake = 0.8;
    t_params.view_angle = 0;
    % down-sampling params
    stride = 1.0;
    N = floor(N);
    num_step = floor(N/stride);
    color_list = parula(round(N * 1.1));
    same_pane_num = 8;
    middle_alpha = 0.5;

    % main loop 
    cnt = 1;
    t_params.t = 0.0;
    for ii = 1: num_step 

        k = floor(1 + stride * (ii - 1));
        rc = rot(1, k); rs = rot(2, k);
        sx = s(1, k); sy = s(2, k);
        px = p(1, k); py = p(2, k);
        Fx = F(1, k); Fy = F(2, k);
        scale_Fx = unit_length * Fx / F_max;
        scale_Fy = unit_length * Fy / F_max;

        if same_pane_num > 1
            if cnt == same_pane_num 
                cnt = 1;
                alpha = 1.0;
            elseif cnt == 1
                t_params.t = t_params.t + dt_fake;
                cnt = cnt + 1;
                alpha = 1.0;
            else 
                cnt = cnt + 1;
                alpha = middle_alpha;
            end
        else
            t_params.t = t_params.t + dt_fake;
            alpha = 1.0;
        end

        % draw final goal
        final = draw_linechart(rc_final, rs_final, sx_final, sy_final, vertices_box, final_plot_params, t_params);
        % draw slider
        slider_plot_params.color = color_list(k, :);
        slider_plot_params.alpha = alpha;
        slider = draw_linechart(rc, rs, sx, sy, vertices_box, slider_plot_params, t_params);
        % draw pusher
        pusher_plot_params.color = color_list(k, :);
        pusher_plot_params.alpha = alpha;
        [wx, wy] = S2W(px, py, rc, rs, sx, sy);
        pusher = draw_point(wx, wy, pusher_plot_params, t_params);
        % draw force
        tmp = sqrt(scale_Fx^2 + scale_Fy^2);
        if tmp > 1e-2
            [scale_Fx, scale_Fy] = S2W(scale_Fx, scale_Fy, rc, rs, 0, 0);
            force = draw_quiver(wx, wy, scale_Fx, scale_Fy, force_plot_params, t_params);
        end
    end

    axis equal;
    set(gca, 'Color', 'white');
    hold off;
    axis off;
    set(gcf, 'PaperPositionMode', 'auto');
    set(gca, 'LooseInset', get(gca, 'TightInset'));
    exportgraphics(gcf, img_filename, 'Resolution', 300);
end

if if_draw_gif
    f = figure('Position', [100, 100, 800, 600]);
    hold on; 
    axis equal;
    axis off;
    set(gca, 'Color', 'white');
    % set(gcf, 'PaperPositionMode', 'auto');
    % set(gca, 'LooseInset', get(gca, 'TightInset'));

    x_min = -s_max * 1.2; x_max = s_max * 1.2; 
    y_min = -s_max * 1.2; y_max = s_max * 1.2;
    xlim([x_min, x_max]);  
    ylim([y_min, y_max]);
    % final goal params
    final_plot_params.lw = 1.0; final_plot_params.color = [0.6, 0.2, 0.0]; final_plot_params.alpha =1.0;
    % pusher params
    pusher_plot_params.s = 30; pusher_plot_params.alpha = 1.0;
    % slider params
    slider_plot_params.lw = 2.0; slider_plot_params.alpha = 1.0;
    % force params
    force_plot_params.lw = 2.0; force_plot_params.color = [1, 0, 0]; force_plot_params.arrow_size = 1.5;
    unit_length = 0.15;
    % time params
    dt_fake = 1.0;
    t_params.view_angle = 0;
    % down-sampling params
    stride = 1.0;
    N = floor(params.N);
    num_step = floor(N/stride);
    color_list = parula(round(N * 1.1));
    % alpha decay
    alpha_decay = 0.8;
    alpha_trunc = floor(num_step/3);

    % draw final goal
    final = draw_linechart(rc_final, rs_final, sx_final, sy_final, vertices_box, final_plot_params);

    slider_cellarr = cell(N, 1);
    pusher_cellarr = cell(N, 1);

    % main loop 
    frame_count = 0; 
    fps = 10;
    slow_rate = 5;
    loop_length = max(1/fps/dt/slow_rate, 1);
    for ii = 1: num_step 
        k = floor(1 + stride * (ii - 1));
        rc = rot(1, k); rs = rot(2, k);
        sx = s(1, k); sy = s(2, k);
        px = p(1, k); py = p(2, k);
        Fx = F(1, k); Fy = F(2, k);
        scale_Fx = unit_length * Fx / F_max;
        scale_Fy = unit_length * Fy / F_max;

        % draw slider
        slider_plot_params.color = color_list(k, :);
        slider = draw_linechart(rc, rs, sx, sy, vertices_box, slider_plot_params);
        slider_cellarr{k} = slider;
        % draw pusher
        pusher_plot_params.color = color_list(k, :);
        [wx, wy] = S2W(px, py, rc, rs, sx, sy);
        pusher = draw_point(wx, wy, pusher_plot_params);
        pusher_cellarr{k} = pusher;
        % draw force
        tmp = sqrt(scale_Fx^2 + scale_Fy^2);
        [scale_Fx, scale_Fy] = S2W(scale_Fx, scale_Fy, rc, rs, 0, 0);
        force = draw_quiver(wx, wy, scale_Fx, scale_Fy, force_plot_params);

         % set alpha decay
         for i = 1: ii
            if (ii - i) >= alpha_trunc
                alpha = 0.0;
            else
                alpha = alpha_decay^(ii - i);
            end
            kk = floor(1 + stride * (i - 1));
            color = color_list(kk, :);
            set_color_linechart(slider_cellarr{kk}, color, alpha);
            set_color_point(pusher_cellarr{kk}, color, alpha);
        end

        drawnow;
        pause(dt);
        if(mod(k, loop_length)==0)
            frame = getframe(gcf);
            img = frame2im(frame);
            [img_ind, cm] = rgb2ind(img, 256);
            if frame_count == 0
                imwrite(img_ind, cm, gif_filename, 'gif', 'LoopCount', Inf, 'DelayTime', 1/fps);
            elseif k == num_step
                imwrite(img_ind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 2);
            else
                imwrite(img_ind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 1/fps);
            end
            frame_count = frame_count + 1;
        end

        delete(force);
    end
end





%% helper functions
function [px, py] = tune_p(px, py, lam, eta, a, b)
    idx = find(round(lam));
    px = min(a-a/8, max(-a+a/8, px));
    py = min(b-b/8, max(-b+b/8, py));
    if idx == 1
        py = py + eta;
    elseif idx == 2
        px = px + eta;
    elseif idx == 3
        py = py - eta;
    elseif idx == 4
        px = px - eta;
    end
end

function c = draw_circle(sx, sy, r, plot_params, t_params)
    color = plot_params.color;
    alpha = plot_params.alpha;
    lw = plot_params.lw;
    th = linspace(0, 2*pi, 1000);
    x = sx + r * cos(th);
    y = sy + r * sin(th);
    if nargin > 4
        t = t_params.t;
        view_angle = t_params.view_angle;
        x = x + t * cos(view_angle);
        y = y + t * sin(view_angle);
    end
    c = plot(x, y, 'Color', [color, alpha], 'LineWidth', lw);
end

function p = draw_point(sx, sy, plot_params, t_params)
    color = plot_params.color;
    alpha = plot_params.alpha;
    s = plot_params.s;
    if nargin > 3
        t = t_params.t;
        view_angle = t_params.view_angle;
        sx = sx + t * cos(view_angle);
        sy = sy + t * sin(view_angle);
    end
    p = scatter(sx, sy, 'SizeData', s, 'MarkerEdgeColor', color, 'MarkerFaceColor', color, ...
                    'MarkerFaceAlpha', alpha, 'MarkerEdgeAlpha', alpha);
end

function set_color_point(p, color, alpha)
    p.MarkerFaceAlpha = alpha;
    p.MarkerEdgeAlpha = alpha;
    p.MarkerEdgeColor = color;
    p.MarkerFaceColor = color;
end

function l = draw_linechart(rc, rs, sx, sy, vertices, plot_params, t_params)
    color = plot_params.color;
    alpha = plot_params.alpha;
    lw = plot_params.lw;
    % each row of vertices: (x, y)
    for i = 1: size(vertices, 1)
        px = vertices(i, 1); py = vertices(i, 2);
        [wx, wy] = S2W(px, py, rc, rs, sx, sy);
        vertices(i, 1) = wx; vertices(i, 2) = wy;
    end
    if size(vertices, 1) > 2
        vertices(end+1, :) = vertices(1, :);
    end
    if nargin > 6
        t = t_params.t;
        view_angle = t_params.view_angle;
        vertices(:, 1) = vertices(:, 1) + t * cos(view_angle);
        vertices(:, 2) = vertices(:, 2) + t * sin(view_angle);
    end
    l = plot(vertices(:, 1), vertices(:, 2), 'Color', [color, alpha], 'LineWidth', lw);
end

function set_color_linechart(l, color, alpha)
    l.Color = [color, alpha];
end

function q = draw_quiver(sx, sy, scale_x, scale_y, plot_params, t_params)
    lw = plot_params.lw;
    color = plot_params.color;
    arrow_size = plot_params.arrow_size;
    if nargin > 5
        t = t_params.t;
        view_angle = t_params.view_angle;
        sx = sx + t * cos(view_angle);
        sy = sy + t * sin(view_angle);
    end
    q = quiver(sx, sy, scale_x, scale_y, ...
            'AutoScale','off', 'MaxHeadSize', arrow_size, 'LineWidth', lw, 'Color', color);
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