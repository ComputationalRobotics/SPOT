% mode 1:
% input: the order in cone
% output: the two-dimensional coordinate in I

% mode 2:
% input: the two-dimensional coordinate in I
% output: the order in cone

function index = find_loca(order, sizes, mode)
    % Passing sizes = cellfun(@numel, I) directly is more efficient
    
    if mode == 1  % Convert global index to [cell index, local index]
        cellIdx = 1;
        while order > sizes(cellIdx)
            order = order - sizes(cellIdx);
            cellIdx = cellIdx + 1;
        end
        index = [cellIdx, order];
        
    else  % Convert [cell index, local index] to global index
        index = sum(sizes(1:order(1)-1)) + order(2);
    end
end
