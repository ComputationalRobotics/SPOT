% mode 1:
% input: the order in cone
% output: the two-dimensional coordinate in I

% mode 2:
% input: the two-dimensional coordinate in I
% output: the order in cone

function index = find_loca(order, sizes, mode)
    % 直接传入sizes = cellfun(@numel, I)更高效
    
    if mode == 1  % 总序号转为[cell序号, 局部序号]
        cellIdx = 1;
        while order > sizes(cellIdx)
            order = order - sizes(cellIdx);
            cellIdx = cellIdx + 1;
        end
        index = [cellIdx, order];
        
    else  % [cell序号, 局部序号]转为总序号
        index = sum(sizes(1:order(1)-1)) + order(2);
    end
end
