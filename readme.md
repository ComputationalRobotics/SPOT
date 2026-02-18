<p align="center">
  <h1 align="center">Sparse Polynomial Optimization Toolbox (SPOT)</h1>
</p>

<p align="center">
  <a href="https://shuchengkang.github.io/"><strong>Shucheng Kang</strong></a><sup>1</sup>
  &nbsp;&nbsp;
  <a href="https://www.linkedin.com/in/guorui-liu-gt/"><strong>Guorui Liu</strong></a><sup>2</sup>
  &nbsp;&nbsp;
  <a href="https://www.linkedin.com/in/haoran-sun-7516b6300/"><strong>Haoran Sun</strong></a><sup>3</sup>
  &nbsp;&nbsp;
  <a href="https://xyxu2033.github.io/"><strong>Xiaoyang Xu</strong></a><sup>4</sup>
  &nbsp;&nbsp;
  <a href="https://hankyang.seas.harvard.edu/"><strong>Heng Yang</strong></a><sup>1</sup>
  <br>
  <sup>1</sup>Harvard University &nbsp;&nbsp; <sup>2</sup>Georgia Tech &nbsp;&nbsp; <sup>3</sup>Fudan University &nbsp;&nbsp; <sup>4</sup>UC Santa Barbara
</p>

<p align="center">
  <em>(Listed in alphabetical order by last name)</em>
</p>

<p align="center">
  <a href="https://computationalrobotics.seas.harvard.edu/project-spot/"><img src="https://img.shields.io/badge/🌐-Website-blue" alt="Website"></a>
  <a href="https://arxiv.org/abs/2502.02829"><img src="https://img.shields.io/badge/📄-Arxiv-red" alt="Arxiv"></a>
</p>

## 📖 About
◊
**SPOT** is a lightweight, high-performance, sparse Moment-SOS Hierarchy conversion package written in C++, with MATLAB and Python interfaces. SPOT is highly inspired by the Julia package [TSSOS](https://github.com/wangjie212/TSSOS).

## 🔥 News

- 🎉 **SPOT** has been accepted by RSS 2025!
- 📦 Introduced **NumPoly** — a lightweight, SymPy-free polynomial builder for the Python interface. See [`Python-examples/test_CSTSS_Python.py`](Python-examples/test_CSTSS_Python.py) for a commented toy example.
- 🚀 We have sped up the **SPOT** Python interface by **10x** on average with **NumPoly**!

## ✨ Features

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

## 🐍 Python Installation and Usage

Create a new conda environment:

```bash
conda create -n spot python=3.10 numpy scipy -y
conda activate spot
conda install -c mosek mosek=10.0 -y
conda install -c conda-forge pybind11 -y
brew install eigen
```

Pybind compilation:

```bash
cd ./SPOT/PYTHON
mkdir build
cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python)
make
```

Then, in the `./Python-examples` folder, you can find examples for sparse polynomial optimization arising from contact-rich planning. A toy example is shown in `./Python-examples/test_CSTSS_Python.py`, where the following polynomial optimization is solved and the minimizer is extracted:

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

- Push Bot: `./Python-examples/pushBot_Python.py`
- Push Box: `./Python-examples/pushBox_Python.py`
- Push Box with Tunnel: `./Python-examples/pushBoxTunnel2_Python.py`
- Push T: `./Python-examples/pushT_Python.py`
- Planar Hand: `./Python-examples/planarHand_Python.py`

## 🛠️ MATLAB Installation and Usage

Please first install [Mosek](https://docs.mosek.com/latest/toolbox/install-interface.html) and [msspoly](https://github.com/spot-toolbox/spotless/tree/master) in MATLAB. Mex file compilation:

```bash
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

## 📝 Citing Our Work

If you use **SPOT** in your research, please consider citing our work:

```bibtex
@inproceedings{kang2025global,
  title={Global contact-Rich planning with sparsity-rich semidefinite relaxations},
  author={Kang, Shucheng and Liu, Guorui and Yang, Heng},
  journal={Robotics: Science and Systems},
  year={2025}
}
@inproceedings{kang2024fast,
  title={Fast and certifiable trajectory optimization},
  author={Kang, Shucheng and Xu, Xiaoyang and Sarva, Jay and Liang, Ling and Yang, Heng},
  journal={International Workshop on the Algorithmic Foundations of Robotics},
  year={2024}
}