close all; clear; 

if_draw_image = false;
if_draw_gif = true;

%% get data
prefix = "2025-01-29_19-04-21";
filename = "data_ordered_YULIN";
% choose the right name
prefix = "/PushBot_MATLAB/" + prefix + "/";
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

%% get hyper-parameters and trajectory
cart_width = 0.1;       
cart_height = 0.05;      
l = params.l;   
d1 = -params.d1;      
d2 = params.d2; 
id = params.id;
a_max = params.a_max;
u_max = params.u_max;
lam1_max = params.lam1_max;
lam2_max = params.lam2_max;    
dt = params.dt; 
N = params.N;    
num_steps = N; 
car_vertices = [
    -0.5 * cart_width, 0.5 * cart_height;
    0.5 * cart_width, 0.5 * cart_height;
    0.5 * cart_width, -0.5 * cart_height;
    -0.5 * cart_width, -0.5 * cart_height;
];
% rescale variables
traj = rescale_sol(v, id, N, a_max, u_max, lam1_max, lam2_max);
rot = traj.rot;
a = traj.a;
u = traj.u;
lam = traj.lam;

%% draw image
if if_draw_image
    figure; hold on;
    x_min = d1 - 0.5; x_max = d2 + 0.5; 
    y_min = -l - 0.01; y_max = l + 0.01;
    wall_width = 0.2;
    % main loop
    stride = 2;
    num_step = floor(N/stride);
    color_list = parula(round(N * 1.1));

    wall_lw = 0.5;
    wall_color = [0, 0, 0, 0.2];
    cart_lw = 1.0;
    pole_lw = 1.0;
    pole_s = 4; 
    force_lw = 1.0;
    force_color = [1, 0, 0, 1];
    force_unit = 1.0;
    force1_unit = [0, 0, force_unit, 0];
    force2_unit = [0, 0, -force_unit, 0];

    dt_fake = 0.1;
    view_angle = pi/3;
    for ii = 1: num_step 
        t = dt_fake * ii;
        wall_left = plot([d1, d1] + t * cos(view_angle), [y_min, y_max] + t * sin(view_angle), 'LineWidth', wall_lw, 'Color', wall_color);
        wall_right = plot([d2, d2] + t * cos(view_angle), [y_min, y_max] + t * sin(view_angle), 'LineWidth', wall_lw, 'Color', wall_color);
        trail = plot([d1, d2] + t * cos(view_angle), [0.0, 0.0] + + t * sin(view_angle), 'LineWidth', wall_lw, 'Color', wall_color);

        k = floor(1 + stride * (ii - 1));
        c = draw_cart(a(k), car_vertices, color_list(k, :), cart_lw, 1.0, true, t, view_angle);
        p = draw_pole(a(k), rot(1, k), rot(2, k), l, color_list(k, :), pole_lw, pole_s, 1.0, true, t, view_angle);

        % add force arrow
        lam1 = lam(1, k); lam2 = lam(2, k);
        if abs(lam1) > 1e-2
            f1 = draw_force(a(k), rot(1, k), rot(2, k), lam1, lam1_max, ...
                l, force1_unit, force_color, force_lw, 1.0, true, t, view_angle);
        end
        if abs(lam2) > 1e-2
            f2 = draw_force(a(k), rot(1, k), rot(2, k), lam2, lam2_max, ...
                l, force2_unit, force_color, force_lw, 1.0, true, t, view_angle);
        end
    end
    axis equal;
    set(gca, 'Color', 'white');
    hold off;
    axis off;
    set(gcf, 'PaperPositionMode', 'auto');
    set(gca, 'LooseInset', get(gca, 'TightInset'));
    % print(img_filename, '-dpng', ['-r', num2str(300)]);
    exportgraphics(gcf, img_filename, 'Resolution', 300);
end



%% draw gif
if if_draw_gif
    f = figure('Position', [100, 100, 800, 600]);
    hold on; 
    axis equal;
    axis off;
    x_min = d1 - 0.5; x_max = d2 + 0.5; 
    y_min = -l - 0.01; y_max = l + 0.01;
    wall_width = 0.5;
    xlim([x_min, x_max]);  
    ylim([y_min, y_max]);
    wall_left = line([d1, d1], [y_min, y_max], 'LineWidth', wall_width, 'Color', [0, 0, 0]);
    wall_right = line([d2, d2], [y_min, y_max], 'LineWidth', wall_width, 'Color', [0, 0, 0]);
    trail = line([d1, d2], [0.0, 0.0], 'LineWidth', wall_width, 'Color', [0, 0, 0]);

    % main loop
    frame_count = 0; 
    fps = 10;
    slow_rate = 5;
    loop_length = max(1/fps/dt/slow_rate, 1);

    stride = 1.0;
    num_step = floor(N/stride);
    color_list = parula(round(N * 1.1));
    cart_lw = 2;
    pole_lw = 2;
    pole_s = 30; 
    alpha_decay = 0.8;
    alpha_trunc = floor(num_step/3);

    force_lw = 2;
    force_color = [1, 0, 0, 1];
    force_unit = 1.0;
    force1_unit = [0, 0, force_unit, 0];
    force2_unit = [0, 0, -force_unit, 0];

    c_cellarr = cell(N, 1);
    p_cellarr = cell(N, 1);

    for i = 1: num_step 
        k = floor(1 + stride * (i - 1));
        c = draw_cart(a(k), car_vertices, color_list(k, :), cart_lw, 1.0, true);
        p = draw_pole(a(k), rot(1, k), rot(2, k), l, color_list(k, :), pole_lw, pole_s, 1.0, true);
        c_cellarr{k} = c;
        p_cellarr{k} = p;

        % add force arrow
        lam1 = lam(1, k); lam2 = lam(2, k);
        if abs(lam1) > 1e-2
            f1 = draw_force(a(k), rot(1, k), rot(2, k), lam1, lam1_max, ...
                l, force1_unit, force_color, force_lw, 1.0, true);
        end
        if abs(lam2) > 1e-2
            f2 = draw_force(a(k), rot(1, k), rot(2, k), lam2, lam2_max, ...
                l, force2_unit, force_color, force_lw, 1.0, true);
        end

        % set alpha decay
        for ii = 1: i 
            if (i - ii) >= alpha_trunc
                alpha = 0.0;
            else
                alpha = alpha_decay^(i - ii);
            end
            kk = floor(1 + stride * (ii - 1));
            color = color_list(kk, :);
            set_color_cart(c_cellarr{kk}, color, alpha);
            set_color_pole(p_cellarr{kk}, color, alpha);
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

        if abs(lam1) > 1e-2
            delete(f1);
        end
        if abs(lam2) > 1e-2
            delete(f2);
        end
    end
end



%% helper functions
function c = draw_cart(a, car_vertices, ...
    color, linewidth, alpha, if_hold_on, t, view_angle)
    car_vertices(:, 1) = car_vertices(:, 1) + a;
    rec_vertex_aug = car_vertices;
    rec_vertex_aug(end+1, :) = rec_vertex_aug(1, :);

    if nargin == 6
        c = plot(rec_vertex_aug(:, 1), rec_vertex_aug(:, 2), 'Color', [color, alpha], 'LineWidth', linewidth);
    else
        rec_vertex_aug(:, 1) = rec_vertex_aug(:, 1) + t * cos(view_angle);
        rec_vertex_aug(:, 2) = rec_vertex_aug(:, 2) + t * sin(view_angle);
        c = plot(rec_vertex_aug(:, 1), rec_vertex_aug(:, 2), 'Color', [color, alpha], 'LineWidth', linewidth);
    end

    if if_hold_on
        hold on;
    end
end

function delete_cart(c)
    delete(c);
end

function set_color_cart(c, color, alpha)
    c.Color = [color, alpha];
end

function p = draw_pole(a, rc, rs, ...
    l, color, linewidth, s, alpha, if_hold_on, t, view_angle)
    x1 = a; x2 = a + l * rs;
    y1 = 0; y2 = -l * rc;

    if nargin > 9
        x1 = x1 + t * cos(view_angle);
        x2 = x2 + t * cos(view_angle);
        y1 = y1 + t * sin(view_angle);
        y2 = y2 + t * sin(view_angle);
    end

    p.line = plot([x1, x2], [y1, y2], 'Color', [color, alpha], 'LineWidth', linewidth);
    p.s1 = scatter(x1, y1, 'SizeData', s, 'MarkerEdgeColor', color, 'MarkerFaceColor', color, ...
                    'MarkerFaceAlpha', alpha, 'MarkerEdgeAlpha', alpha);
    p.s2 = scatter(x2, y2, 'SizeData', s, 'MarkerEdgeColor', color, 'MarkerFaceColor', color, ...
                    'MarkerFaceAlpha', alpha, 'MarkerEdgeAlpha', alpha);

    if if_hold_on
        hold on;
    end
end

function delete_pole(p)
    delete(p.line); delete(p.s1); delete(p.s2);
end

function set_color_pole(p, color, alpha)
    p.line.Color = [color, alpha];
    p.s1.MarkerFaceAlpha = alpha;
    p.s1.MarkerEdgeAlpha = alpha;
    p.s1.MarkerEdgeColor = color;
    p.s1.MarkerFaceColor = color;
    p.s2.MarkerFaceAlpha = alpha;
    p.s2.MarkerEdgeAlpha = alpha;
    p.s2.MarkerEdgeColor = color;
    p.s2.MarkerFaceColor = color;
end

function f = draw_force(a, rc, rs, lam, lam_max, ...
    l, force_unit, color, linewidth, alpha, if_hold_on, t, view_angle)
    force_unit = force_unit * lam / lam_max;
    x2 = a + l * rs;
    y2 = -l * rc;
    force_unit(1) = force_unit(1) + x2;
    force_unit(2) = force_unit(2) + y2;
    if nargin > 11
        force_unit(1) = force_unit(1) + t * cos(view_angle);
        force_unit(2) = force_unit(2) + t * sin(view_angle);
    end
    f = quiver(force_unit(1), force_unit(2), force_unit(3), force_unit(4), ...
            'AutoScale','off', 'MaxHeadSize', 1.0, 'LineWidth', linewidth, 'Color', color);
end

function traj = rescale_sol(sol, id, N, ...
    a_max, u_max, lam1_max, lam2_max)
    v_opt = sol;
    a_opt = zeros(1, N+1); % abandon the last a
    rot_opt = zeros(4, N+1); % abandon the last rs
    u_opt = zeros(1, N);
    lam_opt = zeros(2, N);
    
    % Loop for k = 0:N
    for k = 0:N
        a_opt(k+1) = a_max * v_opt(id("a", k));
        rot_opt(1, k+1) = v_opt(id("rc", k));
        rot_opt(2, k+1) = v_opt(id("rs", k));
        rot_opt(3, k+1) = v_opt(id("fc", k));
        rot_opt(4, k+1) = v_opt(id("fs", k));
    end
    
    % Loop for k = 1:N
    for k = 1:N
        u_opt(k) = u_max * v_opt(id("u", k));
        lam_opt(1, k) = lam1_max * v_opt(id("lam1", k));
        lam_opt(2, k) = lam2_max * v_opt(id("lam2", k));
    end

    traj.a = a_opt; 
    traj.rot = rot_opt;
    traj.u = u_opt;
    traj.lam = lam_opt;
end