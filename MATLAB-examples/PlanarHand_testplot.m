function tmp = PlanarHand_testplot(prefix, if_draw_image, if_draw_gif)

%% get data
filename = "data_ordered_YULIN";
% choose the right name
prefix = "/PlanarHand_MATLAB/" + prefix + "/";
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
disp(N);
dt = params.dt;

% rescale variables
traj = rescale_sol(v, params);

if if_draw_image
    figure; hold on;
    % plot params
    finger_plot_params.circle_lw = 1.0;
    finger_plot_params.line_lw = 1.0;
    finger_plot_params.alpha = 1.0;
    finger_plot_params.color = [0, 0, 0];

    plane_plot_params.circle_lw = 1.0;
    plane_plot_params.line_lw = 1.0;
    plane_plot_params.alpha = 1.0;
    plane_plot_params.color = [0, 0, 0];

    force_plot_params.lw = 1.0; 
    force_plot_params.lamn_color = [1, 0, 0];
    force_plot_params.lamt_color = [1, 0, 0]; 
    force_plot_params.arrow_size = 1.5;
    force_plot_params.unit_length = 0.4;
    % down-sampling params
    stride = 1.0;
    N = floor(N);
    num_step = floor(N/stride);
    color_list = parula(round(N * 1.2));
    same_pane_num = 5;
    middle_alpha = 0.5;
    % time params
    dt_fake_small = 0.18;
    view_angle_small = pi/2 * 0.7;
    dt_fake_large = 0.5;
    view_angle_large = 0.0;

    % main loop 
    cnt = 1;
    dx_small = 0.0;
    dy_small = 0.0;
    dx_large = 0.0;
    dy_large = 0.0;
    t_params.t = 0.0;
    for ii = 1: num_step 

        k = floor(1 + stride * (ii - 1));
        finger_plot_params.color = color_list(k, :);
        plane_plot_params.color = color_list(k, :);

        if same_pane_num > 1
            if cnt == same_pane_num 
                dx_small = dx_small + dt_fake_small * cos(view_angle_small);
                dy_small = dy_small + dt_fake_small * sin(view_angle_small);;
                cnt = 1;
                alpha = 1.0;
            elseif cnt == 1
                dx_large = dx_large + dt_fake_large * cos(view_angle_large);
                dy_large = dy_large + dt_fake_large * sin(view_angle_large);
                dx_small = 0.0;
                dy_small = 0.0;
                cnt = cnt + 1;
                alpha = 1.0;
            else 
                dx_small = dx_small + dt_fake_small * cos(view_angle_small);
                dy_small = dy_small + dt_fake_small * sin(view_angle_small);;
                cnt = cnt + 1;
                alpha = middle_alpha;
            end
        else
            dx_small = dx_small + dt_fake_small * cos(view_angle_small);
            dy_small = dy_small + dt_fake_small * sin(view_angle_small);;
            alpha = 1.0;
        end
        t_params.dx = dx_small + dx_large;
        t_params.dy = dy_small + dy_large;
        finger_plot_params.alpha = alpha;
        plane_plot_params.alpha = alpha;

        % draw plane
        h_plane = draw_plane(traj, k, params, plane_plot_params, t_params);
        % draw left finger 
        h_left_finger = draw_left_finger(traj, k, params, finger_plot_params, t_params);
        % draw right finger
        h_right_finger = draw_right_finger(traj, k, params, finger_plot_params, t_params);
        % draw left force
        h_left_force = draw_left_force(traj, k, params, force_plot_params, t_params);
        % draw right force
        h_right_force = draw_right_force(traj, k, params, force_plot_params, t_params);
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

    x_min = -params.x_max; x_max = params.x_max; 
    y_min = -params.y_max; y_max = params.y_max;
    xlim([x_min, x_max]);  
    ylim([y_min, y_max]);

    % plot params
    finger_plot_params.circle_lw = 1.0;
    finger_plot_params.line_lw = 1.0;
    finger_plot_params.alpha = 1.0;
    finger_plot_params.color = [0, 0, 0];

    plane_plot_params.circle_lw = 1.0;
    plane_plot_params.line_lw = 1.0;
    plane_plot_params.alpha = 1.0;
    plane_plot_params.color = [0, 0, 0];

    force_plot_params.lw = 1.0; 
    force_plot_params.lamn_color = [1, 0, 0];   
    force_plot_params.lamt_color = [1, 0, 0]; 
    force_plot_params.arrow_size = 1.5;
    force_plot_params.unit_length = 0.4;
    % down-sampling params
    stride = 1.0;
    N = floor(N);
    num_step = floor(N/stride);
    color_list = parula(round(N * 1.2));
    same_pane_num = 1;
    middle_alpha = 1.0;
    % time params
    dt_fake_small = 0.0;
    view_angle_small = 0.0;
    dt_fake_large = 0.0;
    view_angle_large = 0.0;

    % main loop 
    cnt = 1;
    t_params.dx = 0;
    t_params.dy = 0;

    % alpha decay
    alpha_decay = 0.8;
    alpha_trunc = 1;

    left_finger_cellarr = cell(N, 1);
    right_finger_cellarr = cell(N, 1);
    plane_cellarr = cell(N, 1);

    % main loop 
    frame_count = 0; 
    fps = 10;
    slow_rate = 5;
    loop_length = max(1/fps/dt/slow_rate, 1);
    for ii = 1: num_step 
        k = floor(1 + stride * (ii - 1));
        finger_plot_params.color = color_list(k, :);
        plane_plot_params.color = color_list(k, :);
        
        % draw plane
        h_plane = draw_plane(traj, k, params, plane_plot_params, t_params);
        % draw left finger 
        h_left_finger = draw_left_finger(traj, k, params, finger_plot_params, t_params);
        % draw right finger
        h_right_finger = draw_right_finger(traj, k, params, finger_plot_params, t_params);
        % draw left force
        h_left_force = draw_left_force(traj, k, params, force_plot_params, t_params);
        % draw right force
        h_right_force = draw_right_force(traj, k, params, force_plot_params, t_params);

        plane_cellarr{k} = h_plane;
        left_finger_cellarr{k} = h_left_finger;
        right_finger_cellarr{k} = h_right_finger;

         % set alpha decay
         for i = 1: ii
            if (ii - i) >= alpha_trunc
                alpha = 0.0;
            else
                alpha = alpha_decay^(ii - i);
            end
            kk = floor(1 + stride * (i - 1));
            color = color_list(kk, :);
            set_color_finger(left_finger_cellarr{kk}, color, alpha);
            set_color_finger(right_finger_cellarr{kk}, color, alpha);
            set_color_plane(plane_cellarr{kk}, color, alpha);
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

        delete_force(h_left_force);
        delete_force(h_right_force);
    end
end

end



%% helper functions
function h = draw_left_force(traj, k, params, plot_params, t_params)
    lamnl = traj.contact.lamnl(k);
    lamtl = traj.contact.lamtl(k);
    lamnl_max = params.lamnl_max; lamtl_max = params.lamtl_max;
    unit_length = plot_params.unit_length;
    lamnl = lamnl / lamnl_max * unit_length;
    lamtl = lamtl / lamtl_max * unit_length;

    xc = traj.circle.pos(1, k);
    yc = traj.circle.pos(2, k);
    l = params.l; r = params.r; H = params.H; Ld = params.Ld; Lu = params.Lu;
    rclu_k = traj.hand.rot_lu(1, k);
    rslu_k = traj.hand.rot_lu(2, k);
    rcld_k = traj.hand.rot_ld(1, k);
    rsld_k = traj.hand.rot_ld(2, k);
    xf = Ld*rcld_k + (2*l+3*r)*rclu_k - H/2; yf = Ld*rsld_k + (2*l+3*r)*rslu_k;

    r = sqrt( (xc - xf)^2 + (yc - yf)^2 );
    rc = (xf - xc) / r;
    rs = (yf - yc) / r;

    [force_x, force_y] = S2W(lamnl, 0, rc, rs, 0, 0);
    lamn_plot_params.lw = plot_params.lw;
    lamn_plot_params.color = plot_params.lamn_color;
    lamn_plot_params.arrow_size = plot_params.arrow_size;
    h.lamn = draw_quiver(xf, yf, force_x, force_y, lamn_plot_params, t_params);

    [force_x, force_y] = S2W(0, lamtl, rc, rs, 0, 0);
    lamt_plot_params.lw = plot_params.lw;
    lamt_plot_params.color = plot_params.lamt_color;
    lamt_plot_params.arrow_size = plot_params.arrow_size;
    h.lamt = draw_quiver(xf, yf, force_x, force_y, lamt_plot_params, t_params);
end

function delete_force(h)
    delete(h.lamn);
    delete(h.lamt);
end

function h = draw_right_force(traj, k, params, plot_params, t_params)
    lamnr = traj.contact.lamnr(k);
    lamtr = traj.contact.lamtr(k);
    lamnr_max = params.lamnr_max; lamtr_max = params.lamtr_max;
    unit_length = plot_params.unit_length;
    lamnr = lamnr / lamnr_max * unit_length;
    lamtr = lamtr / lamtr_max * unit_length;

    xc = traj.circle.pos(1, k);
    yc = traj.circle.pos(2, k);
    l = params.l; r = params.r; H = params.H; Ld = params.Ld; Lu = params.Lu;
    rcru_k = traj.hand.rot_ru(1, k);
    rsru_k = traj.hand.rot_ru(2, k);
    rcrd_k = traj.hand.rot_rd(1, k);
    rsrd_k = traj.hand.rot_rd(2, k);
    xf = Ld*rcrd_k + (2*l+3*r)*rcru_k + H/2; yf = Ld*rsrd_k + (2*l+3*r)*rsru_k;

    r = sqrt( (xc - xf)^2 + (yc - yf)^2 );
    rc = (xf - xc) / r;
    rs = (yf - yc) / r;

    [force_x, force_y] = S2W(lamnr, 0, rc, rs, 0, 0);
    lamn_plot_params.lw = plot_params.lw;
    lamn_plot_params.color = plot_params.lamn_color;
    lamn_plot_params.arrow_size = plot_params.arrow_size;
    h.lamn = draw_quiver(xf, yf, force_x, force_y, lamn_plot_params, t_params);

    [force_x, force_y] = S2W(0, lamtr, rc, rs, 0, 0);
    lamt_plot_params.lw = plot_params.lw;
    lamt_plot_params.color = plot_params.lamt_color;
    lamt_plot_params.arrow_size = plot_params.arrow_size;
    h.lamt = draw_quiver(xf, yf, force_x, force_y, lamt_plot_params, t_params);
end

function q = draw_quiver(sx, sy, scale_x, scale_y, plot_params, t_params)
    lw = plot_params.lw;
    color = plot_params.color;
    arrow_size = plot_params.arrow_size;
    if nargin > 5
        sx = sx + t_params.dx;
        sy = sy + t_params.dy;
    end
    q = quiver(sx, sy, scale_x, scale_y, ...
            'AutoScale','off', 'MaxHeadSize', arrow_size, 'LineWidth', lw, 'Color', color);
end

function h = draw_plane(traj, k, params, plot_params, t_params)
    R = params.R; 
    x_k = traj.circle.pos(1, k);
    y_k = traj.circle.pos(2, k);
    rc_k = traj.circle.rot(1, k);
    rs_k = traj.circle.rot(2, k);

    xc = x_k; yc = y_k;
    xl_1 = x_k; yl_1 = y_k;
    xl_2 = x_k + R * rc_k; yl_2 = y_k + R * rs_k;
    xl_3 = x_k - R * rs_k; yl_3 = y_k + R * rc_k;
    
    color = plot_params.color;
    circle_lw = plot_params.circle_lw;
    line_lw = plot_params.line_lw;
    alpha = plot_params.alpha;

    circle_plot_params.lw = circle_lw;
    circle_plot_params.color = color;
    circle_plot_params.alpha = alpha;
    line_plot_params.lw = line_lw;
    line_plot_params.color = color;
    line_plot_params.alpha = alpha;

    c1 = draw_circle(xc, yc, R, circle_plot_params, t_params); h.c1 = c1;
    l1 = draw_line(xl_1, yl_1, xl_2, yl_2, line_plot_params, t_params); h.l1 = l1;
    l2 = draw_line(xl_1, yl_1, xl_3, yl_3, line_plot_params, t_params); h.l2 = l2;
end

function h = draw_left_finger(traj, k, params, plot_params, t_params)
    l = params.l; r = params.r; H = params.H; Ld = params.Ld; Lu = params.Lu;
    rclu_k = traj.hand.rot_lu(1, k);
    rslu_k = traj.hand.rot_lu(2, k);
    rcld_k = traj.hand.rot_ld(1, k);
    rsld_k = traj.hand.rot_ld(2, k);

    xc_1 = (l+r)*rcld_k - H/2; yc_1 = (l+r) * rsld_k;
    xc_2 = (2*l+3*r)*rcld_k - H/2; yc_2 = (2*l+3*r)*rsld_k;
    xc_3 = Ld*rcld_k + (l+r)*rclu_k - H/2; yc_3 = Ld*rsld_k + (l+r)*rslu_k;
    xc_4 = Ld*rcld_k + (2*l+3*r)*rclu_k - H/2; yc_4 = Ld*rsld_k + (2*l+3*r)*rslu_k;

    xl_1 = -H/2; yl_1 = 0.0;
    xl_2 = Ld*rcld_k - H/2; yl_2 = Ld*rsld_k;
    xl_3 = Ld*rcld_k + Lu*rclu_k - H/2; yl_3 = Ld*rsld_k + Lu*rslu_k;
    
    color = plot_params.color;
    circle_lw = plot_params.circle_lw;
    line_lw = plot_params.line_lw;
    alpha = plot_params.alpha;

    circle_plot_params.lw = circle_lw;
    circle_plot_params.color = color;
    circle_plot_params.alpha = alpha;
    line_plot_params.lw = line_lw;
    line_plot_params.color = color;
    line_plot_params.alpha = alpha;

    c1 = draw_circle(xc_1, yc_1, r, circle_plot_params, t_params); h.c1 = c1; 
    c2 = draw_circle(xc_2, yc_2, r, circle_plot_params, t_params); h.c2 = c2;
    c3 = draw_circle(xc_3, yc_3, r, circle_plot_params, t_params); h.c3 = c3;
    c4 = draw_circle(xc_4, yc_4, r, circle_plot_params, t_params); h.c4 = c4;
    l1 = draw_line(xl_1, yl_1, xl_2, yl_2, line_plot_params, t_params); h.l1 = l1;
    l2 = draw_line(xl_2, yl_2, xl_3, yl_3, line_plot_params, t_params); h.l2 = l2;
end

function set_color_plane(h, color, alpha)
    set_color_circle(h.c1, color, alpha);
    set_color_line(h.l1, color, alpha);
    set_color_line(h.l2, color, alpha);
end

function h = draw_right_finger(traj, k, params, plot_params, t_params)
    l = params.l; r = params.r; H = params.H; Ld = params.Ld; Lu = params.Lu;
    rcru_k = traj.hand.rot_ru(1, k);
    rsru_k = traj.hand.rot_ru(2, k);
    rcrd_k = traj.hand.rot_rd(1, k);
    rsrd_k = traj.hand.rot_rd(2, k);

    xc_1 = (l+r)*rcrd_k + H/2; yc_1 = (l+r) * rsrd_k;
    xc_2 = (2*l+3*r)*rcrd_k + H/2; yc_2 = (2*l+3*r)*rsrd_k;
    xc_3 = Ld*rcrd_k + (l+r)*rcru_k + H/2; yc_3 = Ld*rsrd_k + (l+r)*rsru_k;
    xc_4 = Ld*rcrd_k + (2*l+3*r)*rcru_k + H/2; yc_4 = Ld*rsrd_k + (2*l+3*r)*rsru_k;

    xl_1 = H/2; yl_1 = 0.0;
    xl_2 = Ld*rcrd_k + H/2; yl_2 = Ld*rsrd_k;
    xl_3 = Ld*rcrd_k + Lu*rcru_k + H/2; yl_3 = Ld*rsrd_k + Lu*rsru_k;
    
    color = plot_params.color;
    circle_lw = plot_params.circle_lw;
    line_lw = plot_params.line_lw;
    alpha = plot_params.alpha;

    circle_plot_params.lw = circle_lw;
    circle_plot_params.color = color;
    circle_plot_params.alpha = alpha;
    line_plot_params.lw = line_lw;
    line_plot_params.color = color;
    line_plot_params.alpha = alpha;

    c1 = draw_circle(xc_1, yc_1, r, circle_plot_params, t_params); h.c1 = c1; 
    c2 = draw_circle(xc_2, yc_2, r, circle_plot_params, t_params); h.c2 = c2;
    c3 = draw_circle(xc_3, yc_3, r, circle_plot_params, t_params); h.c3 = c3;
    c4 = draw_circle(xc_4, yc_4, r, circle_plot_params, t_params); h.c4 = c4;
    l1 = draw_line(xl_1, yl_1, xl_2, yl_2, line_plot_params, t_params); h.l1 = l1;
    l2 = draw_line(xl_2, yl_2, xl_3, yl_3, line_plot_params, t_params); h.l2 = l2;
end

function set_color_finger(h, color, alpha)
    set_color_circle(h.c1, color, alpha);
    set_color_circle(h.c2, color, alpha);
    set_color_circle(h.c3, color, alpha);
    set_color_circle(h.c4, color, alpha);
    set_color_line(h.l1, color, alpha);
    set_color_line(h.l2, color, alpha);
end

function l = draw_line(x1, y1, x2, y2, plot_params, t_params)
    color = plot_params.color;
    alpha = plot_params.alpha;
    lw = plot_params.lw;

    if nargin > 5
        x1 = x1 + t_params.dx;
        x2 = x2 + t_params.dx;
        y1 = y1 + t_params.dy;
        y2 = y2 + t_params.dy;
    end

    l = plot([x1, x2], [y1, y2], 'Color', [color, alpha], 'LineWidth', lw);
end

function set_color_line(l, color, alpha)
    l.Color = [color, alpha];
end

function c = draw_circle(sx, sy, r, plot_params, t_params)
    color = plot_params.color;
    alpha = plot_params.alpha;
    lw = plot_params.lw;
    th = linspace(0, 2*pi, 1000);
    x = sx + r * cos(th);
    y = sy + r * sin(th);
    if nargin > 4
        x = x + t_params.dx;
        y = y + t_params.dy;
    end
    c = plot(x, y, 'Color', [color, alpha], 'LineWidth', lw);
end

function set_color_circle(c, color, alpha)
    c.Color = [color, alpha];
end

function p = draw_point(sx, sy, plot_params, t_params)
    color = plot_params.color;
    alpha = plot_params.alpha;
    s = plot_params.s;
    if nargin > 3
        sx = sx + t_params.dx;
        sy = sy + t_params.dy;
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
        vertices(:, 1) = vertices(:, 1) + t_params.dx;
        vertices(:, 2) = vertices(:, 2) + t_params.dy;
    end
    l = plot(vertices(:, 1), vertices(:, 2), 'Color', [color, alpha], 'LineWidth', lw);
end

function set_color_linechart(l, color, alpha)
    l.Color = [color, alpha];
end

function [wx, wy] = S2W(px, py, rc, rs, sx, sy)
    wx = sx + px * rc - py * rs;
    wy = sy + px * rs + py * rc; 
end

function traj = rescale_sol(v, params)
    var_start_dict = params.var_start_dict;
    N = params.N;
    short_list = 0: N-1;
    long_list = 0: N; 

    % "long" variables
    circle.pos = zeros(2, N+1);
    circle.rot = zeros(2, N+1);
    hand.rot_ru = zeros(2, N+1);
    hand.rot_rd = zeros(2, N+1);
    hand.rot_lu = zeros(2, N+1);
    hand.rot_ld = zeros(2, N+1);
    for k = long_list
        circle.pos(1, k+1) = v(var_start_dict("x") + k) * params.x_max;
        circle.pos(2, k+1) = v(var_start_dict("y") + k) * params.y_max;
        circle.rot(1, k+1) = v(var_start_dict("rc") + k);
        circle.rot(2, k+1) = v(var_start_dict("rs") + k);

        hand.rot_ru(1, k+1) = v(var_start_dict("rcru") + k);
        hand.rot_ru(2, k+1) = v(var_start_dict("rsru") + k);
        hand.rot_rd(1, k+1) = v(var_start_dict("rcrd") + k);
        hand.rot_rd(2, k+1) = v(var_start_dict("rsrd") + k);
        hand.rot_lu(1, k+1) = v(var_start_dict("rclu") + k);
        hand.rot_lu(2, k+1) = v(var_start_dict("rslu") + k);
        hand.rot_ld(1, k+1) = v(var_start_dict("rcld") + k);
        hand.rot_ld(2, k+1) = v(var_start_dict("rsld") + k);
    end

    % short variables
    circle.rot_vel = zeros(2, N);
    hand.rot_vel_ru = zeros(2, N);
    hand.rot_vel_rd = zeros(2, N);
    hand.rot_vel_lu = zeros(2, N);
    hand.rot_vel_ld = zeros(2, N);
    hand.pos_r = zeros(2, N);
    hand.pos_l = zeros(2, N);
    hand.pos_vel_r = zeros(2, N);
    hand.pos_vel_l = zeros(2, N);
    contact.lamnr = zeros(1, N);
    contact.lamtr = zeros(1, N);
    contact.lamnl = zeros(1, N);
    contact.lamtl = zeros(1, N);
    contact.dr = zeros(1, N);
    contact.dl = zeros(1, N);
    contact.vrelr = zeros(1, N);
    contact.vrell = zeros(1, N);
    for k = short_list
        circle.rot_vel(1, k+1) = v(var_start_dict("fc") + k);
        circle.rot_vel(2, k+1) = v(var_start_dict("fs") + k);

        hand.rot_vel_ru(1, k+1) = v(var_start_dict("fcru") + k);
        hand.rot_vel_ru(2, k+1) = v(var_start_dict("fsru") + k);
        hand.rot_vel_rd(1, k+1) = v(var_start_dict("fcrd") + k);
        hand.rot_vel_rd(2, k+1) = v(var_start_dict("fsrd") + k);
        hand.rot_vel_lu(1, k+1) = v(var_start_dict("fclu") + k);
        hand.rot_vel_lu(2, k+1) = v(var_start_dict("fslu") + k);
        hand.rot_vel_ld(1, k+1) = v(var_start_dict("fcld") + k);
        hand.rot_vel_ld(2, k+1) = v(var_start_dict("fsld") + k);

        hand.pos_r(1, k+1) = v(var_start_dict("xr") + k) * params.xr_max;
        hand.pos_r(2, k+1) = v(var_start_dict("yr") + k) * params.yr_max;
        hand.pos_l(1, k+1) = v(var_start_dict("xl") + k) * params.xl_max;
        hand.pos_l(2, k+1) = v(var_start_dict("yl") + k) * params.yl_max;
        hand.pos_vel_r(1, k+1) = v(var_start_dict("vxr") + k) * params.vxr_max;
        hand.pos_vel_r(2, k+1) = v(var_start_dict("vyr") + k) * params.vyr_max;
        hand.pos_vel_l(1, k+1) = v(var_start_dict("vxl") + k) * params.vxl_max;
        hand.pos_vel_l(2, k+1) = v(var_start_dict("vyl") + k) * params.vyl_max;

        contact.lamnr(1, k+1) = v(var_start_dict("lamnr") + k) * params.lamnr_max;
        contact.lamtr(1, k+1) = v(var_start_dict("lamtr") + k) * params.lamtr_max;
        contact.lamnl(1, k+1) = v(var_start_dict("lamnl") + k) * params.lamnl_max;
        contact.lamtl(1, k+1) = v(var_start_dict("lamtl") + k) * params.lamtl_max;
        contact.dr(1, k+1) = v(var_start_dict("dr") + k) * params.dr_max;
        contact.dl(1, k+1) = v(var_start_dict("dl") + k) * params.dl_max;
        contact.vrelr(1, k+1) = v(var_start_dict("vrelr") + k) * params.vrelr_max;
        contact.vrell(1, k+1) = v(var_start_dict("vrell") + k) * params.vrell_max;
    end

    traj.circle = circle;
    traj.hand = hand;
    traj.contact = contact;
end