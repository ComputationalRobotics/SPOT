% psd_idx, row, col can all be vectors. psd_idx is the position in the PSD cone, row and col are coordinates, returns the xiaoyang representation of these elements
function rpt = find_rpt(psd_idx, row, col, aux_info)
    size = length(psd_idx);
    rpt = zeros(size, 2 * aux_info.kappa); % Can also be stored directly as key

    % Extract these outside the loop to avoid repeated extraction inside the function
    tI = aux_info.ts_info;
    mon_rpt = aux_info.mon_rpt;
    mon_rpt_g = aux_info.mon_rpt_g;
    cI = aux_info.cliques;

    tI_sizes = cellfun(@numel, tI);

    for i = 1:size
        tmp = find_loca(psd_idx(i), tI_sizes, 1);
        j = tmp(1); k = tmp(2);
        % Must extract the moment part; if converted j exceeds cI length, the block is in the localizing matrix (invalid)
        if j > length(cI)
            disp('Wrong index!');
        end
        rpt(i, :) = mon_rpt{j}{tI{j}{k}(row(i)), tI{j}{k}(col(i))};
    end 
end