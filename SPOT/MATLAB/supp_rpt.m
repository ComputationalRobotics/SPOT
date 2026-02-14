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
    
    % Preallocate array
    seqs = zeros(size(p, 1), d);
    
    % Compute nonzero elements for each row
    [rows, cols] = find(p);
    
    % Use cumulative sum to compute index positions
    cumulative_p = cumsum(p, 2);
    
    % Preallocate cell array to store sequences for each row
    sequences = cell(size(p, 1), 1);
    
    % Vectorized operation using arrayfun
    for i = 1:size(p, 1)
        % Create sequence for the current row
        current_seq = zeros(1, cumulative_p(i, end));
        idx = 1;
        
        % Use logical indexing and repmat
        for j = find(p(i,:))
            rep_count = p(i,j);
            current_seq(idx:idx+rep_count-1) = x_vars_index(j);
            idx = idx + rep_count;
        end
        
        % Right-align padding
        seqs(i, d-length(current_seq)+1:d) = current_seq;
    end

    % seqs = sort(seqs, 2); 
end