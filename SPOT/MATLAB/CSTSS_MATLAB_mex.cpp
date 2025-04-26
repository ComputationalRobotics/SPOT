#include <memory>
#include <iostream>
#include <cassert>
#include <string>

#include "mex.h"
#include "matrix.h"
#include "mat.h"

#include "gen_test.h"  
#include "Polys.h"     
#include "Timer.h"

// Check cell array and return number of cells
int CheckCellArray(const mxArray *array) {
    if (!mxIsCell(array)) {
        mexErrMsgIdAndTxt("MATLAB:CheckCellArray:notCell", "Input must be a cell array of matrices.");
    }
    size_t numCells = mxGetNumberOfElements(array);
    return numCells;
}

// Function to convert an mxArray to an Eigen matrix
Eigen::MatrixXd mxArrayToEigenMatrixDouble(const mxArray *array) {
    if (!mxIsDouble(array) || mxIsComplex(array) || mxGetNumberOfDimensions(array) != 2) {
        mexErrMsgIdAndTxt("MATLAB:mxArrayToEigen:invalidInput", "Each input must be a real double 2D matrix.");
    }

    // Get dimensions of the matrix
    size_t m = mxGetM(array);
    size_t n = mxGetN(array);

    size_t numElements = n * m;
    // Handle empty vector case
    if (numElements == 0) {
        return Eigen::MatrixXd(); // Return an empty Eigen::MatrixXd
    }

    // Get a pointer to the mxArray data
    double *data = mxGetPr(array);

    // Map the MATLAB matrix to an Eigen matrix (no data copy)
    Eigen::Map<Eigen::MatrixXd> eigenMatrix(data, m, n);

    // Create a copy of the Eigen matrix
    return Eigen::MatrixXd(eigenMatrix);
}

// Function to convert an mxArray to an Eigen vector (column or row)
Eigen::VectorXd mxArrayToEigenVectorDouble(const mxArray *array) {
    if (!mxIsDouble(array) || mxIsComplex(array) || mxGetNumberOfDimensions(array) != 2) {
        mexErrMsgIdAndTxt("MATLAB:mxArrayToEigenRowVector:invalidInput", "Input must be a real double 2D matrix.");
    }

    size_t rows = mxGetM(array);
    size_t cols = mxGetN(array);

    size_t numElements = rows * cols;
    // Handle empty vector case
    if (numElements == 0) {
        return Eigen::VectorXd(); // Return an empty Eigen::VectorXd
    }

    if (rows != 1 && cols != 1) {
        mexErrMsgIdAndTxt("MATLAB:mxArrayToEigenRowVector:notVector", "Input must be a row or column vector.");
    }

    double *data = mxGetPr(array);

    // If it's a row vector, map directly as an Eigen::VectorXd
    if (rows == 1) {
        Eigen::Map<Eigen::VectorXd> eigenRowVector(data, cols);
        return Eigen::VectorXd(eigenRowVector); // Return a standalone Eigen row vector
    }

    // If it's a column vector, map and transpose
    Eigen::Map<Eigen::VectorXd> eigenColumnVector(data, rows);
    return Eigen::VectorXd(eigenColumnVector.transpose()); // Convert to row vector
}

// Function to convert a general vector mxArray to std::vector<int>
std::vector<int> mxArrayToStdVectorInt(const mxArray *array) {
    // Validate input
    if (!mxIsNumeric(array) || mxIsComplex(array)) {
        mexErrMsgIdAndTxt("MATLAB:mxArrayToStdVectorInt:invalidInput", "Input must be a real numeric array.");
    }

    // Get dimensions
    size_t rows = mxGetM(array);
    size_t cols = mxGetN(array);

    // Ensure it's a vector (1*n or n*1)
    if (rows != 1 && cols != 1) {
        mexErrMsgIdAndTxt("MATLAB:mxArrayToStdVectorInt:notVector", "Input must be a vector.");
    }

    // Determine the number of elements
    size_t numElements = rows * cols;

    // Get the data pointer
    double *data = mxGetPr(array);

    // Convert to std::vector<int>
    std::vector<int> result(numElements);
    for (size_t i = 0; i < numElements; ++i) {
        result[i] = static_cast<int>(data[i]);
    }

    return result;
}


// Function to read an integer input
int mxArrayToInt(const mxArray *array) {
    // Ensure the input is numeric and scalar
    if (!mxIsNumeric(array) || mxIsComplex(array) || mxGetNumberOfElements(array) != 1) {
        mexErrMsgIdAndTxt("MATLAB:mxArrayToInt:invalidInput", "Input must be a real scalar.");
    }

    // Get the scalar value
    double value = mxGetScalar(array);

    // Check if it can be safely converted to an integer
    if (value != static_cast<int>(value)) {
        mexErrMsgIdAndTxt("MATLAB:mxArrayToInt:notInteger", "Input must be an integer.");
    }

    return static_cast<int>(value);
}

// Function to convert mxArray* to std::string
std::string mxArrayToStdString(const mxArray *array) {
    // Ensure the input is a string
    if (!mxIsChar(array)) {
        mexErrMsgIdAndTxt("MATLAB:mxArrayToStdString:invalidInput", "Input must be a string.");
    }

    // Convert the mxArray to a C-style string
    char *cString = mxArrayToString(array);
    if (cString == nullptr) {
        mexErrMsgIdAndTxt("MATLAB:mxArrayToStdString:conversionFailed", "Failed to convert mxArray to string.");
    }

    // Create a std::string from the C-style string
    std::string result(cString);

    // Free the temporary C-style string allocated by mxArrayToString
    mxFree(cString);

    return result;
}

// Function to convert std::vector<int> to mxArray (1D array)
mxArray* vectorStdToMxArray(const std::vector<int>& vec) {
    size_t numElements = vec.size();
    mxArray* array = mxCreateDoubleMatrix(1, numElements, mxREAL);
    double* data = mxGetPr(array);

    for (size_t i = 0; i < numElements; ++i) {
        data[i] = static_cast<double>(vec[i]);
    }
    return array;
}

// Function to convert std::vector<double> to mxArray (1D array)
mxArray* vectorStdToMxArray(const std::vector<double>& vec) {
    size_t numElements = vec.size();
    mxArray* array = mxCreateDoubleMatrix(1, numElements, mxREAL);
    double* data = mxGetPr(array);

    for (size_t i = 0; i < numElements; ++i) {
        data[i] = vec[i];
    }
    return array;
}

// Function to convert Eigen::VectorXi to mxArray (double vector)
mxArray* vectorEigenToMxArray(const Eigen::VectorXi& vec) {
    size_t numElements = vec.size();

    // Create a double column vector in MATLAB
    mxArray* array = mxCreateDoubleMatrix(numElements, 1, mxREAL);
    double* data = mxGetPr(array);

    // Copy data from Eigen::VectorXi to the MATLAB array
    for (size_t i = 0; i < numElements; ++i) {
        data[i] = static_cast<double>(vec[i]);
    }
    return array;
}

// Function to convert Eigen::VectorXd to mxArray (double vector)
mxArray* vectorEigenToMxArray(const Eigen::VectorXd& vec) {
    size_t numElements = vec.size();

    // Create a double column vector in MATLAB
    mxArray* array = mxCreateDoubleMatrix(numElements, 1, mxREAL);
    double* data = mxGetPr(array);

    // Copy data from Eigen::VectorXi to the MATLAB array
    for (size_t i = 0; i < numElements; ++i) {
        data[i] = vec[i];
    }
    return array;
}

// Function to convert Eigen::MatrixXd to mxArray (double matrix)
mxArray* matrixEigenToMxArray(const Eigen::MatrixXd& matrix) {
    size_t rows = matrix.rows();
    size_t cols = matrix.cols();
    mxArray* array = mxCreateDoubleMatrix(rows, cols, mxREAL);
    double* data = mxGetPr(array);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[j * rows + i] = matrix(i, j); // Column-major order
        }
    }

    return array;
}

// Function to convert Eigen::MatrixXi to mxArray (double matrix)
mxArray* matrixEigenToMxArray(const Eigen::MatrixXi& matrix) {
    size_t rows = matrix.rows();
    size_t cols = matrix.cols();
    mxArray* array = mxCreateDoubleMatrix(rows, cols, mxREAL);
    double* data = mxGetPr(array);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[j * rows + i] = static_cast<double>(matrix(i, j)); // Column-major order
        }
    }

    return array;
}

// Function to convert std::vector<Eigen::VectorXi> to mxArray (cell array)
mxArray* vectorStdVectorEigenToMxCellArray(const std::vector<Eigen::VectorXi>& vecs) {
    size_t numCells = vecs.size();
    mxArray* cellArray = mxCreateCellMatrix(1, numCells);

    for (size_t i = 0; i < numCells; ++i) {
        const Eigen::VectorXi& eigenVec = vecs[i];

        // Create a MATLAB array for each Eigen::VectorXi
        mxArray* cell = mxCreateDoubleMatrix(eigenVec.size(), 1, mxREAL);
        double* data = mxGetPr(cell);

        for (int j = 0; j < eigenVec.size(); ++j) {
            data[j] = static_cast<double>(eigenVec[j]);
        }
        mxSetCell(cellArray, i, cell);
    }

    return cellArray;
}

// Function to convert std::vector<Eigen::MatrixXi> to mxArray (cell array)
mxArray* vectorStdMatrixEigenToMxCellArray(const std::vector<Eigen::MatrixXi>& vec) {
    size_t numCells = vec.size();

    // Create a cell array of size 1xN
    mxArray* cellArray = mxCreateCellMatrix(1, numCells);

    // Iterate over the vector and populate the cell array
    for (size_t i = 0; i < numCells; ++i) {
        // Convert each Eigen::MatrixXi to mxArray
        mxArray* matrixArray = matrixEigenToMxArray(vec[i]);

        // Set the mxArray into the cell array
        mxSetCell(cellArray, i, matrixArray);
    }

    return cellArray;
}

// Function to convert std::vector<std::vector<Eigen::VectorXi>> to mxArray (nested cell array)
mxArray* nestedVectorEigenToMxCellArray(const std::vector<std::vector<Eigen::VectorXi>>& nestedVecs) {
    size_t numOuterCells = nestedVecs.size();

    // Create an outer cell array in MATLAB
    mxArray* outerCellArray = mxCreateCellMatrix(1, numOuterCells);

    // Populate the outer cell array
    for (size_t i = 0; i < numOuterCells; ++i) {
        // Convert the inner vector of Eigen::VectorXi to a MATLAB cell array
        mxArray* innerCellArray = vectorStdVectorEigenToMxCellArray(nestedVecs[i]);
        mxSetCell(outerCellArray, i, innerCellArray); // Assign the inner cell array to the outer cell array
    }
    return outerCellArray;
}

// Current input layout:
//      kappa: relaxation order; an integer
//      total_var_num: total variable number; an integer
//      coeff_f: coefficients of the objecitve; a double row vector
//      supp_f: support representation of the objective; an int matrix
//      coeff_g: coefficients of the inequality constraints; a cell array of double row vectors
//      supp_g: support representation of the inequality constraints; a cell array of int matrices
//      coeff_h: coefficients of the equality constraints; a cell array of double row vectors
//      supp_h: support representation of the equality constraints; a cell array of int matrices
//      dg_list: an int vector of "half" degrees of inequality constraints
//      dh_list: an int vector of degrees of equality constraints
//      relax_mode: relax mode; a string
//      cs_mode: CS mode; a string
//      ts_mode: TS mode; a string
//      ts_mom_mode: add first-order moment matrix in each clique during TS for ease of extraction solution; a string
//      ts_eq_mode: use TS for equality constraints in POP; a string
//      cs_cliques: input CS cliques in CS's SELF mode; a cell array of int vectors
//      ts_cliques: input TS cliques in TS's SELF mode; a cell array of cell arraies of int vectors
class INPUT_ID_factory {
    public:
        int kappa;
        int total_var_num;
        int coeff_f;
        int supp_f;
        int coeff_g;
        int supp_g;
        int coeff_h;
        int supp_h;
        int dg_list;
        int dh_list;

        int relax_mode;
        int cs_mode;
        int ts_mode;
        int ts_mom_mode;
        int ts_eq_mode;
        int cs_cliques;
        int ts_cliques;

        INPUT_ID_factory(int offset = 0) {
            int idx = offset;
            this->kappa = idx; idx = idx + 1;
            this->total_var_num = idx; idx = idx + 1;

            this->coeff_f = idx; idx = idx + 1;
            this->supp_f = idx; idx = idx + 1;
            this->coeff_g = idx; idx = idx + 1;
            this->supp_g = idx; idx = idx + 1;
            this->coeff_h = idx; idx = idx + 1;
            this->supp_h = idx; idx = idx + 1;
            this->dg_list = idx; idx = idx + 1;
            this->dh_list = idx; idx = idx + 1;

            this->relax_mode = idx; idx = idx + 1;
            this->cs_mode = idx; idx = idx + 1;
            this->ts_mode = idx; idx = idx + 1;
            this->ts_mom_mode = idx; idx = idx + 1;
            this->ts_eq_mode = idx; idx = idx + 1;
            this->cs_cliques = idx; idx = idx + 1;
            this->ts_cliques = idx; idx = idx + 1;
        }
};

// Current output layout:
//      cs_info: struct of CS information: {cI, c_g, c_h, mon, mon_g, mon_h, n_cI}
//      ts_info: struct of TS information: {tI, tI_size, tI_num, n_tI}
//      moment_info: {A_moment, C_moment, b_moment}
//      sos_info: {A_sos, a_sos, b_sos, c_sos}
class OUTPUT_ID_factory {
    public:
        int cs_info;
        int ts_info;
        int moment_info;
        int sos_info;

        OUTPUT_ID_factory(int offset = 0) {
            int idx = offset;
            this->cs_info = idx; idx = idx + 1;
            this->ts_info = idx; idx = idx + 1;
            this->moment_info = idx; idx = idx + 1;
            this->sos_info = idx; idx = idx + 1;
        }
};


// main interface
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    INPUT_ID_factory INPUT_ID(0);
    OUTPUT_ID_factory OUTPUT_ID(0);

    // create Polys object
    Polys problem;

    // read kappa
    int kappa = mxArrayToInt(prhs[INPUT_ID.kappa]);
    // read total_var_num
    int total_var_num = mxArrayToInt(prhs[INPUT_ID.total_var_num]);
    // read coeff_f
    Eigen::VectorXd coeff_f = mxArrayToEigenVectorDouble(prhs[INPUT_ID.coeff_f]);
    // read supp_f
    Eigen::MatrixXi supp_f = mxArrayToEigenMatrixDouble(prhs[INPUT_ID.supp_f]).cast<int>();
    // read coeff_g 
    std::vector<Eigen::VectorXd> coeff_g;
    int cell_num_coeff_g = CheckCellArray(prhs[INPUT_ID.coeff_g]);
    for (int i = 0; i < cell_num_coeff_g; ++i) {
        mxArray* cell_content = mxGetCell(prhs[INPUT_ID.coeff_g], i);
        coeff_g.push_back( std::move(mxArrayToEigenVectorDouble(cell_content)) );
    }
    // read supp_g
    std::vector<Eigen::MatrixXi> supp_g;
    int cell_num_supp_g = CheckCellArray(prhs[INPUT_ID.supp_g]);
    assert(cell_num_coeff_g == cell_num_supp_g);
    for (int i = 0; i < cell_num_supp_g; ++i) {
        mxArray* cell_content = mxGetCell(prhs[INPUT_ID.supp_g], i);
        supp_g.push_back( std::move(mxArrayToEigenMatrixDouble(cell_content).cast<int>() ) );
    }
    // read coeff_h
    std::vector<Eigen::VectorXd> coeff_h;
    int cell_num_coeff_h = CheckCellArray(prhs[INPUT_ID.coeff_h]);
    for (int i = 0; i < cell_num_coeff_h; ++i) {
        mxArray* cell_content = mxGetCell(prhs[INPUT_ID.coeff_h], i);
        coeff_h.push_back( std::move(mxArrayToEigenVectorDouble(cell_content)) );
    }
    // read supp_h
    std::vector<Eigen::MatrixXi> supp_h;
    int cell_num_supp_h = CheckCellArray(prhs[INPUT_ID.supp_h]);
    assert(cell_num_coeff_h == cell_num_supp_h);
    for (int i = 0; i < cell_num_supp_h; ++i) {
        mxArray* cell_content = mxGetCell(prhs[INPUT_ID.supp_h], i);
        supp_h.push_back( std::move(mxArrayToEigenMatrixDouble(cell_content).cast<int>() ) );
    }
    // read dg_list
    std::vector<int> dg_list = mxArrayToStdVectorInt(prhs[INPUT_ID.dg_list]);
    assert(dg_list.size() == cell_num_coeff_g);
    // read dh_list
    std::vector<int> dh_list = mxArrayToStdVectorInt(prhs[INPUT_ID.dh_list]);
    assert(dh_list.size() == cell_num_coeff_h);

    // initialize problem
    problem.setObjective(supp_f, coeff_f, total_var_num, kappa);
    problem.setInequalities(supp_g, coeff_g, dg_list);
    problem.setEqualities(supp_h, coeff_h, dh_list);
    // problem.printInfo();

    // read relax_mode
    std::string relax_mode = mxArrayToStdString(prhs[INPUT_ID.relax_mode]);
    std::cout << "relax mode: " << relax_mode << std::endl;
    // read cs_mode
    std::string cs_mode = mxArrayToStdString(prhs[INPUT_ID.cs_mode]);
    std::cout << "cs mode: " << cs_mode << std::endl;
    // read ts_mode 
    std::string ts_mode = mxArrayToStdString(prhs[INPUT_ID.ts_mode]);
    std::cout << "ts mode: " << ts_mode << std::endl;
    // read ts_mom_mode
    std::string ts_mom_mode = mxArrayToStdString(prhs[INPUT_ID.ts_mom_mode]);
    std::cout << "ts mom mode: " << ts_mom_mode << std::endl;
    // read ts_eq_mode
    std::string ts_eq_mode = mxArrayToStdString(prhs[INPUT_ID.ts_eq_mode]);
    std::cout << "ts eq mode: " << ts_eq_mode << std::endl;
    // read cs_cliques
    if (cs_mode == "SELF") {
        int cell_num_cs_cliques = CheckCellArray(prhs[INPUT_ID.cs_cliques]);
        for (int i = 0; i < cell_num_cs_cliques; ++i) {
            mxArray* cell_content = mxGetCell(prhs[INPUT_ID.cs_cliques], i);
            problem.cliques.push_back( std::move(mxArrayToEigenVectorDouble(cell_content).cast<int>() ) );
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
    
    // return cs_info
    const char* cs_info_names[] = {"cI", "c_g", "c_h", "mon", "mon_g", "mon_h", "n_cI"};
    mxArray* cs_info_struct = mxCreateStructMatrix(1, 1, 7, cs_info_names);
    mxSetField(cs_info_struct, 0, "cI", vectorStdVectorEigenToMxCellArray(problem.cI));
    mxSetField(cs_info_struct, 0, "c_g", vectorStdToMxArray(problem.c_g));
    mxSetField(cs_info_struct, 0, "c_h", vectorStdToMxArray(problem.c_h));
    mxSetField(cs_info_struct, 0, "mon", vectorStdMatrixEigenToMxCellArray(problem.mon));
    mxSetField(cs_info_struct, 0, "mon_g", vectorStdMatrixEigenToMxCellArray(problem.mon_g));
    mxSetField(cs_info_struct, 0, "mon_h", vectorStdMatrixEigenToMxCellArray(problem.mon_h));
    mxArray* n_cI_mex = mxCreateDoubleScalar(static_cast<double>(problem.n_cI));
    mxSetField(cs_info_struct, 0, "n_cI", n_cI_mex);
    plhs[OUTPUT_ID.cs_info] = cs_info_struct;

    // return ts_info
    const char* ts_info_names[] = {"tI", "tI_num", "tI_size", "n_tI"};
    mxArray* ts_info_struct = mxCreateStructMatrix(1, 1, 4, ts_info_names);
    mxSetField(ts_info_struct, 0, "tI", nestedVectorEigenToMxCellArray(problem.tI));
    mxSetField(ts_info_struct, 0, "tI_num", vectorStdToMxArray(problem.tI_num));
    mxSetField(ts_info_struct, 0, "tI_size", vectorStdToMxArray(problem.tI_size));
    mxArray* n_tI_mex = mxCreateDoubleScalar(static_cast<double>(problem.n_tI));
    mxSetField(ts_info_struct, 0, "n_tI", n_tI_mex);
    plhs[OUTPUT_ID.ts_info] = ts_info_struct;

    // return moment_info
    const char* moment_info_names[] = {"A_moment", "C_moment", "b_moment"};
    mxArray* moment_info_struct = mxCreateStructMatrix(1, 1, 3, moment_info_names);
    mxSetField(moment_info_struct, 0, "A_moment", matrixEigenToMxArray(problem.A_moment));
    mxSetField(moment_info_struct, 0, "C_moment", matrixEigenToMxArray(problem.C_moment));
    mxSetField(moment_info_struct, 0, "b_moment", vectorEigenToMxArray(problem.b_moment));
    plhs[OUTPUT_ID.moment_info] = moment_info_struct;

    // return sos_info
    const char* sos_info_names[] = {"A_sos", "a_sos", "b_sos", "c_sos"};
    mxArray* sos_info_struct = mxCreateStructMatrix(1, 1, 4, sos_info_names);
    mxSetField(sos_info_struct, 0, "A_sos", matrixEigenToMxArray(problem.A_sos));
    mxSetField(sos_info_struct, 0, "a_sos", matrixEigenToMxArray(problem.a_sos));
    mxSetField(sos_info_struct, 0, "b_sos", matrixEigenToMxArray(problem.b_sos));
    mxSetField(sos_info_struct, 0, "c_sos", vectorEigenToMxArray(problem.c_sos));
    plhs[OUTPUT_ID.sos_info] = sos_info_struct;

    return;
}

