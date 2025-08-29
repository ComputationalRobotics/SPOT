pathinfo = dictionary();

% change the following three package paths to your own paths
pathinfo("mosek") = "/Users/hankyang/mosek";
pathinfo("msspoly") = "/Users/hankyang/Documents/MATLAB/spotless";
pathinfo("spot") = "/Users/hankyang/Documents/MATLAB/SPOT/SPOT/MATLAB";

keys = pathinfo.keys;
for i = 1: length(keys)
    key = keys(i);
    addpath(genpath(pathinfo(key)));
end