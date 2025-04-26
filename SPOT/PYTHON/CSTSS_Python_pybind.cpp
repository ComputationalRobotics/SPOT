#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>    
#include <Eigen/Dense>
#include <iostream>

#include "gen_test.h"  
#include "Polys.h"     
#include "Timer.h"

namespace py = pybind11;

Eigen::MatrixXd array_to_eigen_MatrixXd(const py::array_t<double> &arr)
{
    // Request buffer information from the NumPy array
    auto buf = arr.request();

    // Ensure the input is 2D
    if (buf.ndim != 2) {
        throw std::runtime_error("array_to_eigen: input array must be 2D");
    }

    const ssize_t rows = buf.shape[0];
    const ssize_t cols = buf.shape[1];

    // Create an Eigen::MatrixXd of the appropriate size
    Eigen::MatrixXd mat(rows, cols);

    // Pointer to the raw NumPy data (double*)
    double *ptr = static_cast<double*>(buf.ptr);

    // Because NumPy is (by default) row-major and Eigen is (by default) column-major,
    // we copy element by element. This is the most general approach and handles
    // non-contiguous arrays as well.

    // Compute how many "double steps" we need to move in memory
    // when we move one row down or one column to the right.
    const ssize_t rowStride = buf.strides[0] / static_cast<ssize_t>(sizeof(double));
    const ssize_t colStride = buf.strides[1] / static_cast<ssize_t>(sizeof(double));

    for (ssize_t r = 0; r < rows; ++r) {
        for (ssize_t c = 0; c < cols; ++c) {
            mat(r, c) = ptr[r * rowStride + c * colStride];
        }
    }

    return mat;
}

Eigen::VectorXd array_to_eigen_VectorXd(const py::array_t<double> &arr)
{
    // 1. Request buffer information
    auto buf = arr.request();

    // 2. Check that the array is 1D
    if (buf.ndim != 1) {
        throw std::runtime_error("np_array_to_vectorxd: input array must be 1D");
    }

    // 3. Get the length of the vector
    const ssize_t n = buf.shape[0];

    // 4. Create Eigen::VectorXd of size n
    Eigen::VectorXd vec(n);

    // 5. Pointer to the raw data
    double *ptr = static_cast<double*>(buf.ptr);

    // 6. Because NumPy can have arbitrary strides, compute the stride in “double” units
    const ssize_t stride = buf.strides[0] / static_cast<ssize_t>(sizeof(double));

    // 7. Copy data element-by-element
    for (ssize_t i = 0; i < n; ++i) {
        vec(i) = ptr[i * stride];
    }

    return vec;
}

// ------------------- Output Struct Definitions -------------------
struct CSInfo {
    std::vector<Eigen::VectorXi> cI;
    std::vector<int> c_g;
    std::vector<int> c_h;
    std::vector<Eigen::MatrixXi> mon;
    std::vector<Eigen::MatrixXi> mon_g;
    std::vector<Eigen::MatrixXi> mon_h;
    int n_cI;
};

struct TSInfo {
    std::vector<std::vector<Eigen::VectorXi>> tI;
    std::vector<int> tI_num;
    std::vector<int> tI_size;
    int n_tI;
};

struct MomentInfo {
    Eigen::MatrixXd A_moment;
    Eigen::MatrixXd C_moment;
    Eigen::VectorXd b_moment;
};

struct SOSInfo {
    Eigen::MatrixXd A_sos;
    Eigen::MatrixXd b_sos;
    Eigen::MatrixXd a_sos;
    Eigen::VectorXi c_sos;
};

/**
 * Example function that takes:
 *   1) Two lists of 2D NumPy arrays (list1, list2)
 *   2) Converts each list to a std::vector<Eigen::MatrixXd>
 *   3) Prints out the shapes
 */
std::tuple<CSInfo, TSInfo, MomentInfo, SOSInfo> my_function(
    int kappa, int total_var_num,
    const py::array_t<double>& py_coeff_f, const py::array_t<double>& py_supp_f,
    const std::vector<py::array_t<double>>& py_coeff_g, const std::vector<py::array_t<double>>& py_supp_g,
    const std::vector<py::array_t<double>>& py_coeff_h, const std::vector<py::array_t<double>>& py_supp_h,
    const py::array_t<double>& py_dg_list, const py::array_t<double>& py_dh_list,
    const std::string& relax_mode,
    const std::string& cs_mode,
    const std::string& ts_mode, 
    const std::string& ts_mom_mode,
    const std::string& ts_eq_mode,
    const std::vector<py::array_t<double>>& py_cs_cliques, const std::vector<py::array_t<double>>& py_ts_cliques
) {
    // read coeff_f
    Eigen::VectorXd coeff_f = array_to_eigen_VectorXd(py_coeff_f);
    // read supp_f
    Eigen::MatrixXd supp_f_double = array_to_eigen_MatrixXd(py_supp_f);
    Eigen::MatrixXi supp_f = supp_f_double.cast<int>();
    // read coeff_g
    std::vector<Eigen::VectorXd> coeff_g;
    coeff_g.reserve(py_coeff_g.size());
    for (const auto& arr : py_coeff_g) {
        coeff_g.push_back(array_to_eigen_VectorXd(arr));
    }
    // read supp_g 
    std::vector<Eigen::MatrixXi> supp_g;
    supp_g.reserve(py_supp_g.size());
    for (const auto& arr : py_supp_g) {
        Eigen::MatrixXd tmp = array_to_eigen_MatrixXd(arr);
        supp_g.push_back(tmp.cast<int>());
    }
    // read coeff_h
    std::vector<Eigen::VectorXd> coeff_h;
    coeff_h.reserve(py_coeff_h.size());
    for (const auto& arr : py_coeff_h) {
        coeff_h.push_back(array_to_eigen_VectorXd(arr));
    }
    // read supp_h 
    std::vector<Eigen::MatrixXi> supp_h;
    supp_h.reserve(py_supp_h.size());
    for (const auto& arr : py_supp_h) {
        Eigen::MatrixXd tmp = array_to_eigen_MatrixXd(arr);
        supp_h.push_back(tmp.cast<int>());
    }
    // read dg_list
    Eigen::VectorXd dg_list_double = array_to_eigen_VectorXd(py_dg_list);
    Eigen::VectorXi dg_list_int = dg_list_double.cast<int>();
    std::vector<int> dg_list(dg_list_int.data(), dg_list_int.data() + dg_list_int.size());
    // read dh_list
    Eigen::VectorXd dh_list_double = array_to_eigen_VectorXd(py_dh_list);
    Eigen::VectorXi dh_list_int = dh_list_double.cast<int>();
    std::vector<int> dh_list(dh_list_int.data(), dh_list_int.data() + dh_list_int.size());

    // create Polys object
    Polys problem;

    // initialize problem
    problem.setObjective(supp_f, coeff_f, total_var_num, kappa);
    problem.setInequalities(supp_g, coeff_g, dg_list);
    problem.setEqualities(supp_h, coeff_h, dh_list);
    // problem.printInfo();

    if (cs_mode == "SELF") {
        for (const auto& arr : py_cs_cliques) {
            Eigen::VectorXd tmp = array_to_eigen_VectorXd(arr);
            problem.cliques.push_back(tmp.cast<int>());
        }
    }
    // read ts_cliques
    if (ts_mode == "SELF") {
        // TODO: add SELF mode in TS
    }

    Timer timer;

    // Generate cI
    timer.start();
    problem.Gen_cI(cs_mode);
    timer.stop("Generate cI");
    
    // assign constraints
    timer.start();
    problem.Assign_Constraints();
    timer.stop("Assign Constraints");

    //Record Constraints
    timer.start();
    problem.Record_monomials();
    timer.stop("Record Monomials");

    // Construct C
    timer.start();
    problem.Construct_C();
    timer.stop("Construct C");
    
    // Term Sparsity
    timer.start();
    problem.Gen_tI(ts_mode, ts_mom_mode);
    timer.stop("Generate tI");

    // Moment Conversion
    if (relax_mode == "MOMENT") {
        timer.start();
        problem.Moment_Conversion(ts_eq_mode);
        timer.stop("Moment Conversion");
    }
    
    // SOS Conversion
    if (relax_mode == "SOS") {
        timer.start();
        problem.SOS_Conversion(ts_eq_mode);
        timer.stop("SOS Conversion");
    }

    // return output
    CSInfo cs_info;
    TSInfo ts_info;
    MomentInfo moment_info;
    SOSInfo sos_info;

    // return cs_info
    cs_info.cI = problem.cI;
    cs_info.c_g = problem.c_g;
    cs_info.c_h = problem.c_h;
    cs_info.mon = problem.mon;
    cs_info.mon_g = problem.mon_g;
    cs_info.mon_h = problem.mon_h;
    cs_info.n_cI = problem.n_cI;

    // return ts_info
    ts_info.tI = problem.tI;
    ts_info.tI_num = problem.tI_num;
    ts_info.tI_size = problem.tI_size;
    ts_info.n_tI = problem.n_tI;

    // return moment_info
    moment_info.A_moment = problem.A_moment;
    moment_info.b_moment = problem.b_moment;
    moment_info.C_moment = problem.C_moment;

    // return sos_info
    sos_info.A_sos = problem.A_sos;
    sos_info.b_sos = problem.b_sos;
    sos_info.a_sos = problem.a_sos;
    sos_info.c_sos = problem.c_sos;

    return std::make_tuple(cs_info, ts_info, moment_info, sos_info);
}

/**
 * Create the Python module using pybind11.
 */
PYBIND11_MODULE(CSTSS_Python_pybind, m)
{
    m.doc() = "SPOT Python binding";

    // Expose CSInfo to Python
    py::class_<CSInfo>(m, "CSInfo")
        .def(py::init<>())
        .def_readwrite("cI", &CSInfo::cI)  
        .def_readwrite("c_g", &CSInfo::c_g)      
        .def_readwrite("c_h", &CSInfo::c_h)   
        .def_readwrite("mon", &CSInfo::mon)
        .def_readwrite("mon_g", &CSInfo::mon_g)
        .def_readwrite("mon_h", &CSInfo::mon_h)
        .def_readwrite("n_cI", &CSInfo::n_cI);

    // Expose TSInfo to Python
    py::class_<TSInfo>(m, "TSInfo")
        .def(py::init<>())
        .def_readwrite("tI", &TSInfo::tI)  
        .def_readwrite("tI_num", &TSInfo::tI_num)      
        .def_readwrite("tI_size", &TSInfo::tI_size)   
        .def_readwrite("n_tI", &TSInfo::n_tI);

    // Expose MomentInfo to Python
    py::class_<MomentInfo>(m, "MomentInfo")
        .def(py::init<>())
        .def_readwrite("A_moment", &MomentInfo::A_moment)  
        .def_readwrite("b_moment", &MomentInfo::b_moment)      
        .def_readwrite("C_moment", &MomentInfo::C_moment);

    // Expose SOSInfo to Python
    py::class_<SOSInfo>(m, "SOSInfo")
        .def(py::init<>())
        .def_readwrite("A_sos", &SOSInfo::A_sos)  
        .def_readwrite("b_sos", &SOSInfo::b_sos)      
        .def_readwrite("a_sos", &SOSInfo::a_sos)
        .def_readwrite("c_sos", &SOSInfo::c_sos);

    m.def("my_function", &my_function, 
          "A function that converts lists of 2D NumPy arrays into std::vector<Eigen::MatrixXd>");
}
