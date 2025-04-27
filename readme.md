# Sparse Polynomial Optimization Toolbox (SPOT)

## [Website](https://computationalrobotics.seas.harvard.edu/project-spot/)|[Arxiv](https://arxiv.org/abs/2502.02829)

## About

**SPOT** is a lightweight, high-performance, sparse Moment-SOS Hierarchy conversion package written in C++, with MATLAB and Python interfaces. SPOT is highly inspired by the Julia package [TSSOS](https://github.com/wangjie212/TSSOS).

## News

- **SPOT** has been accepted by RSS 2025!

## Features

**SPOT** tries to solve the following polynomial optimization with sparse Moment-SOS Hierarchy:
$$
\min_{\mathbf{x}} f(\mathbf{x})
$$

$$
\text{subject to } g_i(\mathbf{x}) \ge 0, \ i \in \mathcal{I}
$$

$$
h_j(\mathbf{x}) = 0, \ j \in \mathcal{E}
$$

where $f, g_i, h_j$ are all polynomials. Options in SPOT interface: 

- `relax_mode`: 
  - `"SOS"`: SOS relaxation
  - `"MOMENT"`: moment relaxation 
- `cs_mode` (correlative sparsity pattern, CS):
  - `"MF"`: Use minimal edge-filling heuristics in CS
  - `"MD"`: Use minimal degree heuristics in CS
  - `"NON"`:  No CS 
  - `"SELF"`: User-defined cliques
    - In MATLAB, please define the CS cliques in `params.cliques`
    - In Python, please define the CS cliques in `params["cliques"]`
- `ts_mode` (term sparsity pattern, TS)
  - `"MF"`: Use minimal edge-filling heuristics in TS
  - `"MD"`: Use minimal degree heuristics in TS
  - `"NON"`:  No TS
- `ts_mom_mode` (add additional degree-1 moment matrix when use TS, inspired by TSSOS)
  - `"USE"`: Add degree-1 moment matrix
  - `"NON"`: Do not add degree-1 moment matrix
- `ts_eq_mode` ("partial term sparsity" for equality constraints)
  - `"USE"`: Use partial term sparsity
  - `"NON"`: Do not use partial term sparsity
- `if_solve` (use Mosek to solve SDP)
  - `true`: Solve the SDP
  - `false`: Only conversion, do not solve the SDP

## MATLAB Installation and Usage

Please first install [Mosek](https://docs.mosek.com/latest/toolbox/install-interface.html) and [msspoly](https://github.com/spot-toolbox/spotless/tree/master) in MATLAB. Mex file compilation: 

```
cd ./SPOT/MATLAB 
mkdir build 
cd build 
cmake ..
make 
```

Then, change paths for Mosek, msspoly, and SPOT in `./SPOT/pathinfo/my_path.m`:

```matlab
% change the following three package paths to your own paths
pathinfo("mosek") = "~/ksc/matlab-install/mosek/10.1/toolbox/r2017a";
pathinfo("msspoly") = "~/ksc/matlab-install/spotless";
pathinfo("spot") = "~/ksc/my-packages/SPOT/SPOT/MATLAB";
```

Then, in the `./MATLAB_examples` folder, you can find numerous examples for sparse polynomial optimization arising from contact-rich planning. A toy example is shown in `./MATLAB_examples/test_CSTSS_MATLAB.m`, where the following polynomial optimization is solved and the minimizer is extracted:

$$
\min x_1 + x_2 + x_3
$$

$$
\text{subject to } 2 - x_i \ge 0, \ i = 1, 2, 3 
$$

$$
x_1^2 + x_2^2 = 1, \ x_2^2 + x_3^2 = 1, \ x_2 = 0.5
$$

Five contact-rich planning problems in the paper can be found in: 

- Push Bot: `./MATLAB_examples/PushBot_MATLAB.m`
- Push Box: `./MATLAB_examples/PushBox_MATLAB.m`
- Push Box with Tunnel: `./MATLAB_examples/PushBoxTunnel2_MATLAB.m`
- Push T: `./MATLAB_examples/PushT_MATLAB.m`
- Planar Hand: `./MATLAB_examples/PlanarHand_MATLAB.m`

If you want to reproduce the statistics in the paper, please run `./MATLAB_examples/Test_Gap.m` (It may take a very long time!). The plotting and rendering codes can be found in `./MATLAB_examples/*_goodplot.m` and `./MATLAB_examples/*_visualize.m`, respectively (''*'' represents five dyanamical systems' names).

 ## Python Installation and Usage

Create a new conda environment:

```
conda create -n spot python=3.10 numpy scipy -y
conda activate spot
conda install -c mosek mosek=10.0 -y
```

Pybind compilation:

```
cd ./SPOT/PYTHON 
mkdir build 
cd build 
cmake ..
make 
```

Then, in the `./Python_examples` folder, you can find examples for sparse polynomial optimization arising from contact-rich planning (currently, only Push T task is translated to Python). A toy example is shown in `./Python_examples/test_CSTSS_Python.m`, where the following polynomial optimization is solved and the minimizer is extracted:

$$
\min x_1 + x_2 + x_3
$$

$$
\text{subject to } 2 - x_i \ge 0, \ i = 1, 2, 3 
$$

$$
x_1^2 + x_2^2 = 1, \ x_2^2 + x_3^2 = 1, \ x_2 = 0.5
$$













