% psd_idx, row, col全部可以传入的是向量，psd_idx为整个PSD cone的位置，row和col就是坐标，返回的是这些元素的xiaoyang representation
function rpt = find_rpt(psd_idx, row, col, aux_info)
    size = length(psd_idx);
    rpt = zeros(size, 2 * aux_info.kappa); % 也可以直接存为key

    % 这个部分写的时候拿出来，函数内部不要重复提取
    tI = aux_info.ts_info;
    mon_rpt = aux_info.mon_rpt;
    mon_rpt_g = aux_info.mon_rpt_g;
    cI = aux_info.cliques;

    tI_sizes = cellfun(@numel, tI);

    for i = 1:size
        tmp = find_loca(psd_idx(i), tI_sizes, 1);
        j = tmp(1); k = tmp(2);
        % 提取必须是取的moment的部分，如果换算的j超过cI长度说明block在localizing matrix中，不对
        if j > length(cI)
            disp('Wrong index!');
        end
        rpt(i, :) = mon_rpt{j}{tI{j}{k}(row(i)), tI{j}{k}(col(i))};
    end 
end