% % 初始化哈希表
% hashTable = containers.Map('KeyType', 'uint64', 'ValueType', 'any');
% n = 724;  % 变量范围 [0:n]

% % 存储四元组
% a = 1; b = 2; c = 3; d = 4;
% key = uint64(a + n * b + n^2 * c + n^3 * d);
% hashTable(key) = "Example Value";

% % 检查四元组是否存在
% if isKey(hashTable, key)
%     disp(['Value for key ', num2str(key), ': ', hashTable(key)]);
% else
%     disp('Key not found in hash table.');
% end

% input monomials' representation, output its key
% 先默认d = 2
% function key = get_key(rpt, n)
%     [~, d] = size(rpt); 
%     % 计算 (n+1) 的幂次 [d-1, d-2, ..., 0]
%     powers = (n + 1) .^ (d - 1:-1:0);  
%     % 向量化计算
%     key = rpt * powers';  % 将 rpt 的每行与 powers 对应元素相乘后求和
% end

function key = get_key(rpt, n)
    [m, d] = size(rpt); 
    key = cell(m, 1);
    for i = 1:m
        key_tmp = 0;
        for j = 1:d
            key_tmp = key_tmp + rpt(i, j) * (n + 1) ^ (d - j);
        end
        % key(i) = uint64(key_tmp);
        key{i} = key_tmp;
    end
end

