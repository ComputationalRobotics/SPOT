close all; clear; 

%% get data
prefix = "2025-01-22_02-56-07";
prefix = "/PlanarHand_MATLAB/" + prefix + "/";
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
id = params.id;
var_start_dict = params.var_start_dict;
filename = fig_prefix + img_name;

%% parameters for drawing
draw_params.hand_color = [0, 0, 0];
draw_params.tip_color = [1, 0, 1];
draw_params.hand_linewidth = 1;
draw_params.circle_color = [0, 0, 1];
draw_params.circle_linewidth = 1;

%% get trajectory
traj = rescale_sol(v, params);

dt = 0.1;               
num_steps = size(params.N, 2);   

%% create figure session
dt = params.dt;
frame_count = 0; 
fps = 8;
slow_rate = 5;
loop_length = max(1/fps/dt/slow_rate, 1);
figure;
hold on;
axis equal;
xlim([-params.x_max, params.x_max]);  
ylim([-params.y_max/3, params.y_max + params.R]);

%% test
for k = 0: params.N - 1
    obj_right_finger_list = draw_right_finger(traj, k, params, draw_params);
    obj_left_finger_list = draw_left_finger(traj, k, params, draw_params);
    obj_circle_list = draw_plane(traj, k, params, draw_params);

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

    delete_obj_list(obj_circle_list);
    delete_obj_list(obj_left_finger_list);
    delete_obj_list(obj_right_finger_list);
end




%% helper functions
function obj_list = draw_plane(traj, k, params, draw_params)
    R = params.R; 
    x_k = traj.circle.pos(1, k+1);
    y_k = traj.circle.pos(2, k+1);
    rc_k = traj.circle.rot(1, k+1);
    rs_k = traj.circle.rot(2, k+1);

    xc = x_k; yc = y_k;
    xl_1 = x_k; yl_1 = y_k;
    xl_2 = x_k + R * rc_k; yl_2 = y_k + R * rs_k;
    xl_3 = x_k - R * rs_k; yl_3 = y_k + R * rc_k;
    
    obj_list = [];
    circle_color = draw_params.circle_color;
    circle_linewidth = draw_params.circle_linewidth;
    obj_c = draw_circle(xc, yc, R, circle_color, circle_linewidth); obj_list = [obj_list; obj_c];
    obj_l = draw_line(xl_1, yl_1, xl_2, yl_2, circle_color, circle_linewidth); obj_list = [obj_list; obj_l];
    obj_l = draw_line(xl_1, yl_1, xl_3, yl_3, circle_color, circle_linewidth); obj_list = [obj_list; obj_l];
end

function obj_list = draw_left_finger(traj, k, params, draw_params)
    l = params.l; r = params.r; H = params.H; Ld = params.Ld; Lu = params.Lu;
    rclu_k = traj.hand.rot_lu(1, k+1);
    rslu_k = traj.hand.rot_lu(2, k+1);
    rcld_k = traj.hand.rot_ld(1, k+1);
    rsld_k = traj.hand.rot_ld(2, k+1);

    xc_1 = (l+r)*rcld_k - H/2; yc_1 = (l+r) * rsld_k;
    xc_2 = (2*l+3*r)*rcld_k - H/2; yc_2 = (2*l+3*r)*rsld_k;
    xc_3 = Ld*rcld_k + (l+r)*rclu_k - H/2; yc_3 = Ld*rsld_k + (l+r)*rslu_k;
    xc_4 = Ld*rcld_k + (2*l+3*r)*rclu_k - H/2; yc_4 = Ld*rsld_k + (2*l+3*r)*rslu_k;

    xl_1 = -H/2; yl_1 = 0.0;
    xl_2 = Ld*rcld_k - H/2; yl_2 = Ld*rsld_k;
    xl_3 = Ld*rcld_k + Lu*rclu_k - H/2; yl_3 = Ld*rsld_k + Lu*rslu_k;
    
    obj_list = [];
    hand_color = draw_params.hand_color;
    tip_color = draw_params.tip_color;
    hand_linewidth = draw_params.hand_linewidth;
    obj_c = draw_circle(xc_1, yc_1, r, hand_color, hand_linewidth); obj_list = [obj_list; obj_c];
    obj_c = draw_circle(xc_2, yc_2, r, hand_color, hand_linewidth); obj_list = [obj_list; obj_c];
    obj_c = draw_circle(xc_3, yc_3, r, hand_color, hand_linewidth); obj_list = [obj_list; obj_c];
    obj_c = draw_circle(xc_4, yc_4, r, tip_color, hand_linewidth); obj_list = [obj_list; obj_c];
    obj_l = draw_line(xl_1, yl_1, xl_2, yl_2, hand_color, hand_linewidth); obj_list = [obj_list; obj_l];
    obj_l = draw_line(xl_2, yl_2, xl_3, yl_3, hand_color, hand_linewidth); obj_list = [obj_list; obj_l];
end

function obj_list = draw_right_finger(traj, k, params, draw_params)
    l = params.l; r = params.r; H = params.H; Ld = params.Ld; Lu = params.Lu;
    rcru_k = traj.hand.rot_ru(1, k+1);
    rsru_k = traj.hand.rot_ru(2, k+1);
    rcrd_k = traj.hand.rot_rd(1, k+1);
    rsrd_k = traj.hand.rot_rd(2, k+1);

    xc_1 = (l+r)*rcrd_k + H/2; yc_1 = (l+r) * rsrd_k;
    xc_2 = (2*l+3*r)*rcrd_k + H/2; yc_2 = (2*l+3*r)*rsrd_k;
    xc_3 = Ld*rcrd_k + (l+r)*rcru_k + H/2; yc_3 = Ld*rsrd_k + (l+r)*rsru_k;
    xc_4 = Ld*rcrd_k + (2*l+3*r)*rcru_k + H/2; yc_4 = Ld*rsrd_k + (2*l+3*r)*rsru_k;

    xl_1 = H/2; yl_1 = 0.0;
    xl_2 = Ld*rcrd_k + H/2; yl_2 = Ld*rsrd_k;
    xl_3 = Ld*rcrd_k + Lu*rcru_k + H/2; yl_3 = Ld*rsrd_k + Lu*rsru_k;
    
    obj_list = [];
    hand_color = draw_params.hand_color;
    tip_color = draw_params.tip_color;
    hand_linewidth = draw_params.hand_linewidth;
    obj_c = draw_circle(xc_1, yc_1, r, hand_color, hand_linewidth); obj_list = [obj_list; obj_c];
    obj_c = draw_circle(xc_2, yc_2, r, hand_color, hand_linewidth); obj_list = [obj_list; obj_c];
    obj_c = draw_circle(xc_3, yc_3, r, hand_color, hand_linewidth); obj_list = [obj_list; obj_c];
    obj_c = draw_circle(xc_4, yc_4, r, tip_color, hand_linewidth); obj_list = [obj_list; obj_c];
    obj_l = draw_line(xl_1, yl_1, xl_2, yl_2, hand_color, hand_linewidth); obj_list = [obj_list; obj_l];
    obj_l = draw_line(xl_2, yl_2, xl_3, yl_3, hand_color, hand_linewidth); obj_list = [obj_list; obj_l];
end

function obj_l = draw_line(x1, y1, x2, y2, color, linewidth)
    obj_l = plot([x1, x2], [y1, y2], 'Color', color, 'LineWidth', linewidth);
end

function obj_c = draw_circle(cx, cy, r, color, linewidth)
    th = linspace(0, 2*pi, 1000);
    x = cx + r * cos(th);
    y = cy + r * sin(th);
    obj_c = plot(x, y, 'Color', color, 'LineWidth', linewidth);
end

function delete_obj_list(obj_list)
    for i = 1: length(obj_list)
        delete(obj_list(i));
    end 
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



