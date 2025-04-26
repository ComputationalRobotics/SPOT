% Test
% mode 1: PushBot
% mode 2: PushBox
% mode 3: PushT
% mode 4: Tunnel
% mode 5: PlanarHand

clear; close all; 

addpath("../pathinfo/");
my_path;

% Const
mode = 3;
Test_time = 10;
result_list = zeros(3 * Test_time, 1);
iter_list = zeros(3 * Test_time, 1);
pfeas_list = zeros(3 * Test_time, 1);
dfeas_list = zeros(3 * Test_time, 1);
residual_list = zeros(3 * Test_time, 1);
mosek_list = zeros(3 * Test_time, 1);
operation_list = zeros(3 * Test_time, 1);


if mode == 1
    % initial status
    a_min = 0.0; a_max = 0.95;
    v_min = -1.0; v_max = 1.0;
    th_min = -pi/4; th_max = pi/4;
    dth_min = -3.0; dth_max = 3.0;
    init_matrix = [
        a_min + (a_max - a_min) * rand(1, Test_time);
        v_min + (v_max - v_min) * rand(1, Test_time);
        th_min + (th_max - th_min) * rand(1, Test_time);
        dth_min + (dth_max - dth_min) * rand(1, Test_time);
    ];

    % test
    for i = 1:Test_time
        aux_info = PushBot_test("SOS", "MF", "NON", init_matrix(:, i));
        result_list(i) = aux_info.result;
        iter_list(i) = aux_info.iter;
        pfeas_list(i) = aux_info.pfeas;
        dfeas_list(i) = aux_info.dfeas;
        residual_list(i) = aux_info.max_residual;
        mosek_list(i) = aux_info.mosek_time;
        operation_list(i) = aux_info.time;
        
        clearvars aux_info;
    end
    for i = 1:Test_time
        aux_info = PushBot_test("SOS", "MF", "MAX", init_matrix(:, i));
        result_list(i + Test_time) = aux_info.result;
        iter_list(i + Test_time) = aux_info.iter;
        pfeas_list(i + Test_time) = aux_info.pfeas;
        dfeas_list(i + Test_time) = aux_info.dfeas;
        residual_list(i + Test_time) = aux_info.max_residual;
        mosek_list(i + Test_time) = aux_info.mosek_time;
        operation_list(i + Test_time) = aux_info.time;
        
        clearvars aux_info;
    end
    for i = 1:Test_time
        aux_info = PushBot_test("SOS", "MF", "MF", init_matrix(:, i));
        result_list(i + 2 * Test_time) = aux_info.result;
        iter_list(i + 2 * Test_time) = aux_info.iter;
        pfeas_list(i + 2 * Test_time) = aux_info.pfeas;
        dfeas_list(i + 2 * Test_time) = aux_info.dfeas;
        residual_list(i + 2 * Test_time) = aux_info.max_residual;
        mosek_list(i + 2 * Test_time) = aux_info.mosek_time;
        operation_list(i + 2 * Test_time) = aux_info.time;
        
        clearvars aux_info;
    end
    
    % save
    save("./data/PushBot_test/data.mat", "result_list", "iter_list", "pfeas_list", "dfeas_list", "residual_list", "mosek_list", "operation_list");
end

if mode == 2
    % initial status
    th_min = -pi; th_max = pi;
    sx_min = 0.1; sx_max = 0.4;
    sy_min = 0.1; sy_max = 0.4;
    init_matrix = [
        th_min + (th_max - th_min) * rand(1, Test_time);
        sx_min + (sx_max - sx_min) * rand(1, Test_time);
        sy_min + (sy_max - sy_min) * rand(1, Test_time);
    ];

    % test
    for i = 1:Test_time
        aux_info = PushBox_test("SOS", "MF", "NON", init_matrix(:, i));
        result_list(i) = aux_info.result;
        iter_list(i) = aux_info.iter;
        pfeas_list(i) = aux_info.pfeas;
        dfeas_list(i) = aux_info.dfeas;
        residual_list(i) = aux_info.max_residual;
        mosek_list(i) = aux_info.mosek_time;
        operation_list(i) = aux_info.time;
        
        clearvars aux_info;
    end
    for i = 1:Test_time
        aux_info = PushBox_test("SOS", "MF", "MAX", init_matrix(:, i));
        result_list(i + Test_time) = aux_info.result;
        iter_list(i + Test_time) = aux_info.iter;
        pfeas_list(i + Test_time) = aux_info.pfeas;
        dfeas_list(i + Test_time) = aux_info.dfeas;
        residual_list(i + Test_time) = aux_info.max_residual;
        mosek_list(i + Test_time) = aux_info.mosek_time;
        operation_list(i + Test_time) = aux_info.time;
        
        clearvars aux_info;
    end
    for i = 1:Test_time
        aux_info = PushBox_test("SOS", "MF", "MF", init_matrix(:, i));
        result_list(i + 2 * Test_time) = aux_info.result;
        iter_list(i + 2 * Test_time) = aux_info.iter;
        pfeas_list(i + 2 * Test_time) = aux_info.pfeas;
        dfeas_list(i + 2 * Test_time) = aux_info.dfeas;
        residual_list(i + 2 * Test_time) = aux_info.max_residual;
        mosek_list(i + 2 * Test_time) = aux_info.mosek_time;
        operation_list(i + 2 * Test_time) = aux_info.time;
        
        clearvars aux_info;
    end
    
    % save
    save("./data/PushBox_test/data.mat", "result_list", "iter_list", "pfeas_list", "dfeas_list", "residual_list", "mosek_list", "operation_list");
end

if mode == 3
    % initial status
    th_min = -pi; th_max = pi;
    sx_min = 0.1; sx_max = 0.4;
    sy_min = 0.1; sy_max = 0.4;
    init_matrix = [
        th_min + (th_max - th_min) * rand(1, Test_time);
        sx_min + (sx_max - sx_min) * rand(1, Test_time);
        sy_min + (sy_max - sy_min) * rand(1, Test_time);
    ];

    % test
    for i = 1:Test_time
        aux_info = PushT_test("SOS", "MF", "NON", init_matrix(:, i));
        result_list(i) = aux_info.result;
        iter_list(i) = aux_info.iter;
        pfeas_list(i) = aux_info.pfeas;
        dfeas_list(i) = aux_info.dfeas;
        residual_list(i) = aux_info.max_residual;
        mosek_list(i) = aux_info.mosek_time;
        operation_list(i) = aux_info.time;
        
        clearvars aux_info;
    end
    for i = 1:Test_time
        aux_info = PushT_test("SOS", "MF", "MAX", init_matrix(:, i));
        result_list(i + Test_time) = aux_info.result;
        iter_list(i + Test_time) = aux_info.iter;
        pfeas_list(i + Test_time) = aux_info.pfeas;
        dfeas_list(i + Test_time) = aux_info.dfeas;
        residual_list(i + Test_time) = aux_info.max_residual;
        mosek_list(i + Test_time) = aux_info.mosek_time;
        operation_list(i + Test_time) = aux_info.time;
        
        clearvars aux_info;
    end
    for i = 1:Test_time
        aux_info = PushT_test("SOS", "MF", "MF", init_matrix(:, i));
        result_list(i + 2 * Test_time) = aux_info.result;
        iter_list(i + 2 * Test_time) = aux_info.iter;
        pfeas_list(i + 2 * Test_time) = aux_info.pfeas;
        dfeas_list(i + 2 * Test_time) = aux_info.dfeas;
        residual_list(i + 2 * Test_time) = aux_info.max_residual;
        mosek_list(i + 2 * Test_time) = aux_info.mosek_time;
        operation_list(i + 2 * Test_time) = aux_info.time;
        
        clearvars aux_info;
    end
    
    % save
    save("./data/PushT_test/data.mat", "result_list", "iter_list", "pfeas_list", "dfeas_list", "residual_list", "mosek_list", "operation_list");
end

if mode == 4
    % initial status
    th_min = -pi; th_max = pi;
    sx_min = -0.5; sx_max = -0.4;
    sy_min = -0.4; sy_max = 0.4;
    init_matrix = [
        th_min + (th_max - th_min) * rand(1, Test_time);
        sx_min + (sx_max - sx_min) * rand(1, Test_time);
        sy_min + (sy_max - sy_min) * rand(1, Test_time);
    ];

    % test
    for i = 1:Test_time
        aux_info = PushBoxTunnel2_test("SOS", "SELF", "NON", init_matrix(:, i));
        result_list(i) = aux_info.result;
        iter_list(i) = aux_info.iter;
        pfeas_list(i) = aux_info.pfeas;
        dfeas_list(i) = aux_info.dfeas;
        residual_list(i) = aux_info.max_residual;
        mosek_list(i) = aux_info.mosek_time;
        operation_list(i) = aux_info.time;
        
        clearvars aux_info;
    end
    for i = 1:Test_time
        aux_info = PushBoxTunnel2_test("SOS", "SELF", "MAX", init_matrix(:, i));
        result_list(i + Test_time) = aux_info.result;
        iter_list(i + Test_time) = aux_info.iter;
        pfeas_list(i + Test_time) = aux_info.pfeas;
        dfeas_list(i + Test_time) = aux_info.dfeas;
        residual_list(i + Test_time) = aux_info.max_residual;
        mosek_list(i + Test_time) = aux_info.mosek_time;
        operation_list(i + Test_time) = aux_info.time;
        
        clearvars aux_info;
    end
    for i = 1:Test_time
        aux_info = PushBoxTunnel2_test("SOS", "SELF", "MF", init_matrix(:, i));
        result_list(i + 2 * Test_time) = aux_info.result;
        iter_list(i + 2 * Test_time) = aux_info.iter;
        pfeas_list(i + 2 * Test_time) = aux_info.pfeas;
        dfeas_list(i + 2 * Test_time) = aux_info.dfeas;
        residual_list(i + 2 * Test_time) = aux_info.max_residual;
        mosek_list(i + 2 * Test_time) = aux_info.mosek_time;
        operation_list(i + 2 * Test_time) = aux_info.time;
        
        clearvars aux_info;
    end
    
    % save
    save("./data/PushBoxTunnel2_test/data.mat", "result_list", "iter_list", "pfeas_list", "dfeas_list", "residual_list", "mosek_list", "operation_list");
end

if mode == 5
    % initial status
    R_min = 0.08; R_max = 0.1;
    init_matrix = [
        R_min + (R_max - R_min) * rand(1, Test_time);
    ];

    % test
    for i = 1:Test_time
        aux_info = PlanarHand_test("SOS", "SELF", "NON", init_matrix(:, i));
        result_list(i) = aux_info.result;
        iter_list(i) = aux_info.iter;
        pfeas_list(i) = aux_info.pfeas;
        dfeas_list(i) = aux_info.dfeas;
        residual_list(i) = aux_info.max_residual;
        mosek_list(i) = aux_info.mosek_time;
        operation_list(i) = aux_info.time;

        clearvars aux_info;
    end
    % for i = 1:Test_time
    %     aux_info = PlanarHand_test("SOS", "SELF", "MAX", init_matrix(:, i));
    %     result_list(i + Test_time) = aux_info.result;
    %     iter_list(i + Test_time) = aux_info.iter;
    %     pfeas_list(i + Test_time) = aux_info.pfeas;
    %     dfeas_list(i + Test_time) = aux_info.dfeas;
    %     residual_list(i + Test_time) = aux_info.max_residual;
    %     mosek_list(i + Test_time) = aux_info.mosek_time;
    %     operation_list(i + Test_time) = aux_info.time;

    %     clearvars aux_info;
    % end
    % for i = 1:Test_time
    %     aux_info = PlanarHand_test("SOS", "SELF", "MF", init_matrix(:, i));
    %     result_list(i + 2 * Test_time) = aux_info.result;
    %     iter_list(i + 2 * Test_time) = aux_info.iter;
    %     pfeas_list(i + 2 * Test_time) = aux_info.pfeas;
    %     dfeas_list(i + 2 * Test_time) = aux_info.dfeas;
    %     residual_list(i + 2 * Test_time) = aux_info.max_residual;
    %     mosek_list(i + 2 * Test_time) = aux_info.mosek_time;
    %     operation_list(i + 2 * Test_time) = aux_info.time;

    %     clearvars aux_info;
    % end
    
    % save
    save("./data/PlanarHand_test/data.mat", "result_list", "iter_list", "pfeas_list", "dfeas_list", "residual_list", "mosek_list", "operation_list");
end