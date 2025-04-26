close all; clear; 

%% get data
prefix = "2025-01-18_02-20-05";
prefix = "/PushBot_MATLAB/" + prefix + "/";
data_prefix = "./data" + prefix;
fig_prefix = "./figs" + prefix;

filename = "data_naive_YULIN";
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

cart_width = 0.4;       
cart_height = 0.2;      
pendulum_length = 0.8;   

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

frame_count = 0; 
fps = 10;
slow_rate = 5;
loop_length = max(1/fps/dt/slow_rate,1);

% rescale variables
traj = rescale_sol(v, id, N, a_max, u_max, lam1_max, lam2_max);
rot_iter = traj.rot;
a_iter = traj.a;
u_iter = traj.u;
lam_iter = traj.lam;

figure;
hold on;
axis equal;
xlim([d1 - 1, d2 + 1]);  
ylim([-pendulum_length - 0.5, pendulum_length + 0.5]);

cart = rectangle('Position', [a_iter(1)-cart_width/2, -cart_height/2, cart_width, cart_height], ...
                 'FaceColor', [0 0.5 1], 'EdgeColor', 'k');

pendulum = line([a_iter(1), a_iter(1) + pendulum_length * rot_iter(2, 1)], ...
                [0, -pendulum_length * rot_iter(1, 1)], 'LineWidth', 2, 'Color', 'k');

wall_left = line([d1, d1], [-1.5, 1.5], 'LineWidth', 2, 'Color', 'r');
wall_right = line([d2, d2], [-1.5, 1.5], 'LineWidth', 2, 'Color', 'r');

title('PushBot');
xlabel('x');
ylabel('y');

for k = 1:num_steps
    set(cart, 'Position', [a_iter(k)-cart_width/2, -cart_height/2, cart_width, cart_height]);

    pendulum_x = [a_iter(k), a_iter(k) + pendulum_length * rot_iter(2, k)];
    pendulum_y = [0, -pendulum_length * rot_iter(1, k)];

    set(pendulum, 'XData', pendulum_x, 'YData', pendulum_y);
    
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
