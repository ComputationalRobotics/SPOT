% input: coefficient
% output: sdp result

function [res, mosek_time] = mosek_standard_sdp_test_1(A, C, b, s)

    prob.c = [];
    prob.a = sparse([], [], [], length(b), 0);
    prob.blc = b;
    prob.buc = b;
    prob.bardim = s';

    % barc
    prob.barc.subj = C(:, 1);
    prob.barc.subk = C(:, 3);
    prob.barc.subl = C(:, 2);
    prob.barc.val = C(:, 4);

    % bara
    prob.bara.subi = A(:, 1);
    prob.bara.subj = A(:, 2);
    prob.bara.subk = A(:, 4);
    prob.bara.subl = A(:, 3);
    prob.bara.val = A(:, 5);

    % solve
    tic;
    param.MSK_IPAR_INTPNT_MAX_ITERATIONS = 100;
    [~, res] = mosekopt('minimize info', prob, param);
    mosek_time = toc;

    % % retrieve the result
    % X = {};
    % sum = 0;
    % for i = 1:length(s)
    %     l = s(i);
    %     [r, c] = find(tril(ones(l)));
    %     list = sub2ind([l,l], r, c);
    %     X{i} = zeros(l, l);
    %     X{i}(list) = res.sol.itr.barx(sum + 1: sum + length(list));
    %     X{i} = X{i} + tril(X{i}, -1)';
    %     sum = sum + length(list);
    % end

end

