% input: polynomial
% output: monomial representation

% function [seqs, coeff] = supp_rpt(f, x, d)
%     [x_vars, p, coeff] = decomp(f);
%     [~, x_vars_index] = ismember(x_vars.var, x.var);
    
%     seqs = zeros(size(p, 1), d);
%     for i = 1:size(p, 1)
%         seq = zeros(1, d); index = 1;
%         for j = 1:length(x_vars)
%             seq(1, index:index + p(i, j) - 1) = x_vars_index(j) * ones(1, p(i, j));
%             index = index + p(i, j);
%         end
%         seq = [zeros(1, d - index + 1), seq(1, 1:index - 1)];
%         seqs(i, :) = seq;
%     end
% end

function [seqs, coeff] = supp_rpt(f, x, d)
    [x_vars, p, coeff] = decomp(f);
    [~, x_vars_index] = ismember(x_vars.var, x.var);
    
    % 预分配数组
    seqs = zeros(size(p, 1), d);
    
    % 计算每行的非零元素
    [rows, cols] = find(p);
    
    % 使用累积和计算索引位置
    cumulative_p = cumsum(p, 2);
    
    % 预分配cell数组存储每行的序列
    sequences = cell(size(p, 1), 1);
    
    % 使用arrayfun进行向量化操作
    for i = 1:size(p, 1)
        % 创建当前行的序列
        current_seq = zeros(1, cumulative_p(i, end));
        idx = 1;
        
        % 使用logical indexing和repmat
        for j = find(p(i,:))
            rep_count = p(i,j);
            current_seq(idx:idx+rep_count-1) = x_vars_index(j);
            idx = idx + rep_count;
        end
        
        % 右对齐填充
        seqs(i, d-length(current_seq)+1:d) = current_seq;
    end
end