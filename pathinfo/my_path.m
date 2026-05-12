pathinfo = dictionary();

% change the following three package paths to your own paths
pathinfo("mosek") = "~/mosek";
pathinfo("msspoly") = "~/Documents/MATLAB/spotless";
pathinfo("spot") = fullfile(fileparts(mfilename('fullpath')), '..', 'SPOT', 'MATLAB');

keys = pathinfo.keys;
for i = 1: length(keys)
    key = keys(i);
    addpath(genpath(pathinfo(key)));
end