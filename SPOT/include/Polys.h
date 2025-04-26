#ifndef POLYS_H
#define POLYS_H

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <unordered_map>

class Polys {
public:
    // Data members

    int n = 0;                                     // Number of variables
    int d = 0;                                     // relaxation order
    Eigen::MatrixXi supp_f;                        // Support of the objective polynomial (dense matrix)
    Eigen::VectorXd coeff_f;                       // Coefficients of the objective polynomial (vector)

    std::vector<Eigen::MatrixXi> supp_g;           // Support of inequality constraints (vector of matrices)
    std::vector<Eigen::VectorXd> coeff_g;          // Coefficients of inequality constraints (vector of vectors)
    std::vector<int> dj_g;                         // Degrees of inequality constraints
    int m_ineq = 0;                                // Number of inequality constraints

    std::vector<Eigen::MatrixXi> supp_h;           // Support of equality constraints (vector of matrices)
    std::vector<Eigen::VectorXd> coeff_h;          // Coefficients of equality constraints (vector of vectors)
    std::vector<int> dj_h;                         // Degrees of equality constraints
    int m_eq = 0;                                  // Number of equality constraints

    std::vector<Eigen::VectorXi> cliques;          // self setting cliques
    std::vector<Eigen::VectorXi> cI;               // correlative sparsity cliques    
    std::vector<int> c_g;                          // inequality constraints assign
    std::vector<int> c_h;                          // equality constraints assign
    int n_cI = 0;
    std::string cs_mode;                           

    std::vector<Eigen::MatrixXi> mon;              // monomials representation of moment matrix
    std::vector<Eigen::MatrixXi> mon_g;            // monomials representation of localizing matrix
    std::vector<Eigen::MatrixXi> mon_h;            // monomials representation of equality constraints
    std::vector<Eigen::VectorXi> even_mon;         // odd 0 even 1
    std::vector<Eigen::VectorXi> even_mon_g;       // odd 0 even 1
    std::vector<Eigen::VectorXi> even_mon_h;       // odd 0 even 1
    std::vector<Eigen::VectorXi> even_supp_g;      // odd 0 even 1
    std::vector<Eigen::VectorXi> even_supp_h;      // odd 0 even 1

    std::unordered_map<uint64_t, int> C;           // term sparsity C
    std::vector<std::vector<Eigen::VectorXi> > tI; // term sparsity index
    std::vector<int> tI_num;                       // blocks number of every matrix 
    std::vector<int> tI_size;                      // blocks sizes
    int n_tI = 0;                                  // number of all blocks
    std::string ts_mode;
    std::string ts_mom_mode;
    std::string ts_eq_mode;

    Eigen::MatrixXd A_moment;                      // moment conversion coefficient
    Eigen::MatrixXd C_moment;                      
    Eigen::VectorXd b_moment;       

    Eigen::MatrixXd A_sos;                         // SOS conversion efficient
    Eigen::MatrixXd a_sos;
    Eigen::MatrixXd b_sos; 
    Eigen::VectorXi c_sos;            

    std::vector<std::vector<int>> idx_ineq;                     // {itr_A, i1, i5} coeff_g[i1](i5)
    std::vector<std::vector<int>> idx_eq;                       // {itr_a, i1, i5} coeff_h[i1](i5)

    

    // Constructor
    Polys() = default;

    // Initialize the objective polynomial f
    void setObjective(const Eigen::MatrixXi& supp, const Eigen::VectorXd& coeff, const int& n0, const int& d0);
    // Initialize the inequality constraints g
    void setInequalities(const std::vector<Eigen::MatrixXi>& supp, const std::vector<Eigen::VectorXd>& coeff, const std::vector<int>& degrees);
    // Initialize the equality constraints h
    void setEqualities(const std::vector<Eigen::MatrixXi>& supp, const std::vector<Eigen::VectorXd>& coeff, const std::vector<int>& degrees);

    // Print polynomial information
    void printInfo() const;

    // Operator =
    Polys& operator=(const Polys& other);

    // Correlative Sparsity: Csp Construction and Chordal Extension
    void Gen_cI(const std::string& cs_mode);

    // Assign Constraints
    void Assign_Constraints();

    // Regroup Monomials
    void Record_monomials();

    // Construct C
    void Construct_C();

    // Term Sparsity: Support extension + block closure
    void Gen_tI(const std::string& ts_mode, const std::string& ts_mom_mode);

    // Moment Conversion
    void Moment_Conversion(const std::string& ts_eq_mode);

    // SOS Conversion
    void SOS_Conversion(const std::string& ts_eq_mode);

    // For Real Machine
    void CSTSS_Real(const std::vector<int>& change_idx_g, const std::vector<int>& change_idx_h,
                    const std::vector<Eigen::VectorXd>& change_coeff_g, const std::vector<Eigen::VectorXd>& change_coeff_h, const int if_first);

    // Debug function
    void printdebug(Eigen::MatrixXi& G) const;
    void printdebug2(Eigen::VectorXi& G) const;
    void printtI() const;
    void printkey(std::unordered_map<uint64_t, int> alpha) const;
    void printmoment() const;
    void printsos() const;

    
                                


private:
    // Print matrix and vector information
    void printMatrix(const Eigen::MatrixXi& supp, const Eigen::VectorXd& coeff) const;
    // Print constraints (supports g and h)
    void printConstraints(const std::vector<Eigen::MatrixXi>& supp, const std::vector<Eigen::VectorXd>& coeff, const std::vector<int>& degrees) const;

    // Csp Construction and Chordal Extension
    Eigen::MatrixXi fill_edges(Eigen::MatrixXi& G, const std::vector<int>& neighbors);
    Eigen::MatrixXi maximal_chordal_extension(const Eigen::MatrixXi& G);
    Eigen::MatrixXi greedy_chordal_extension(const Eigen::MatrixXi& G, std::vector<int>& order);
    Eigen::MatrixXi minimal_fill_chordal_extension(const Eigen::MatrixXi& G, std::vector<int>& order);
    std::vector<Eigen::VectorXi> find_max_cliques(const Eigen::MatrixXi& G, const std::vector<int>& order, int mode);
    Eigen::MatrixXi csp_construct();
    std::vector<Eigen::VectorXi> chordal_extension(const Eigen::MatrixXi& G, int mode);

    // Record Monomials
    Eigen::MatrixXi build_union_vector(int n, int d, const Eigen::VectorXi& x_vars);
    Eigen::MatrixXi generate_all_sequences(int n, int d);
    Eigen::VectorXi find_next_monomial(int n, const Eigen::VectorXi& J, int d);

    // Construct C
    bool check_even(const Eigen::VectorXi& vec);
    std::vector<uint64_t> get_key(const Eigen::MatrixXi& rpt, int n);
    Eigen::MatrixXi generate_2N(int n, int d);

    // support extension + block closure
    Eigen::RowVectorXi merge_sort(const Eigen::RowVectorXi& row1, const Eigen::RowVectorXi& row2);
    // notice that B_eq is also matrix not vector
    Eigen::MatrixXi hs_support_extension_rpt(const std::vector<Eigen::MatrixXi>& mon_rpt, const std::unordered_map<uint64_t, int>& C, const Eigen::MatrixXi& supp_rpt_g, int idx_g, int mode);
    int find_root(int i, std::vector<int>& parent);
    Eigen::MatrixXi block_closure(const Eigen::MatrixXi& B);
    std::vector<Eigen::VectorXi> find_blocks(const Eigen::MatrixXi& B);

    // Moment and SOS
    std::vector<int> find_loca_1(int order, std::vector<int>& sizes);
    int find_loca_2(std::vector<int> order, std::vector<int>& sizes);

    // Real Machine
    void SOS_Conversion_Real(const std::string& ts_eq_mode);

};

// for mex file debug
inline void pl(int line) {
    std::cout << "Called from line " << line << std::endl;
}

#endif