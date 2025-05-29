#include "Polys.h"
#include "Timer.h"
#include <algorithm>
#include <set>
#include <unordered_map>    
#include <numeric> 
#include <queue>


// Initialize the objective polynomial f
void Polys::setObjective(const Eigen::MatrixXi& supp, const Eigen::VectorXd& coeff, const int& n0, const int& d0) {
    supp_f = supp;
    coeff_f = coeff;
    n = n0;
    d = d0;
}

// Initialize the inequality constraints g
void Polys::setInequalities(const std::vector<Eigen::MatrixXi>& supp, const std::vector<Eigen::VectorXd>& coeff, const std::vector<int>& degrees) {
    supp_g = supp;
    coeff_g = coeff;
    dj_g = degrees;
    m_ineq = supp.size();
}

// Initialize the equality constraints h
void Polys::setEqualities(const std::vector<Eigen::MatrixXi>& supp, const std::vector<Eigen::VectorXd>& coeff, const std::vector<int>& degrees) {
    supp_h = supp;
    coeff_h = coeff;
    dj_h = degrees;
    m_eq = supp.size();
}

// Print polynomial information
void Polys::printInfo() const {
    std::cout << "Objective Polynomial:\n";
    printMatrix(supp_f, coeff_f);

    std::cout << "\nInequality Constraints (" << m_ineq << "):\n";
    printConstraints(supp_g, coeff_g, dj_g);

    std::cout << "\nEquality Constraints (" << m_eq << "):\n";
    printConstraints(supp_h, coeff_h, dj_h);
}

// Print matrix and vector information
void Polys::printMatrix(const Eigen::MatrixXi& supp, const Eigen::VectorXd& coeff) const {
    for (int i = 0; i < coeff.size(); ++i) {
        std::cout << "Coeff: " << coeff(i) << ", Supp: ";
        for (int j = 0; j < supp.cols(); ++j) {
            std::cout << supp(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

// Print constraints (supports g and h)
void Polys::printConstraints(const std::vector<Eigen::MatrixXi>& supp, const std::vector<Eigen::VectorXd>& coeff, const std::vector<int>& degrees) const {
    for (size_t i = 0; i < supp.size(); ++i) {
        std::cout << "Constraint " << i + 1 << " (Degree: " << degrees[i] << "):\n";
        for (int j = 0; j < coeff[i].size(); ++j) {
            std::cout << "Coeff: " << coeff[i](j) << ", Supp: ";
            for (int k = 0; k < supp[i].cols(); ++k) {
                std::cout << supp[i](j, k) << " ";
            }
            std::cout << std::endl;
        }
    }
}

Polys& Polys::operator=(const Polys& other) {
    if (this != &other) {  
        n = other.n;
        supp_f = other.supp_f;
        coeff_f = other.coeff_f;

        supp_g = other.supp_g;
        coeff_g = other.coeff_g;
        dj_g = other.dj_g;
        m_ineq = other.m_ineq;

        supp_h = other.supp_h;
        coeff_h = other.coeff_h;
        dj_h = other.dj_h;
        m_eq = other.m_eq;

        cI = other.cI;
    }
    return *this;
}


//Generate cI
 void Polys::Gen_cI(const std::string& cs_mode) {
    Timer timer;

    timer.start();
    Eigen::MatrixXi G = csp_construct();
    timer.stop("csp_construct");

    if (cs_mode == "SELF") {
        cI = cliques;
    } else if (cs_mode == "MAX") {
        cI = chordal_extension(G, 1);
    } else if (cs_mode == "MD") {
        cI = chordal_extension(G, 2);
    } else if (cs_mode == "MF") {
        cI = chordal_extension(G, 3);
    } else if (cs_mode == "NON") {
        cI.push_back(Eigen::VectorXi::LinSpaced(n, 1, n));
    } else {
        std::cerr << "Error: Unknown cs_mode '" << cs_mode << "'" << std::endl;
    }

    n_cI = cI.size();
 }

// Assign Constraints
void Polys::Assign_Constraints() {

    //inequality constraints
    // int max_size = std::max(10 * m_ineq, 10000); // assume max ten times
    std::vector<Eigen::MatrixXi> supp_g_re;
    std::vector<Eigen::VectorXd> coeff_g_re;
    std::vector<int> dj_g_re;
    std::vector<int> c_g_re;
    int m_ineq_re = 0;


    for (int i = 0; i < m_ineq; ++i) {
        std::vector<int> nz_g;
        for (int row = 0; row < supp_g[i].rows(); ++row) {
            for (int col = 0; col < supp_g[i].cols(); ++col) {
                if (supp_g[i](row, col) != 0) {
                    nz_g.push_back(supp_g[i](row, col));
                }
            }
        }
        

        for (size_t j = 0; j < cI.size(); ++j) {
            bool is_subset = true;
            for (int var : nz_g) {
                if ((cI[j].array() == var).count() == 0) {
                    is_subset = false;
                    break;
                }
            }
            

            if (is_subset) {
                supp_g_re.push_back(supp_g[i]);
                coeff_g_re.push_back(coeff_g[i]);
                dj_g_re.push_back(dj_g[i]);
                c_g_re.push_back(j);
                m_ineq_re++;
            }
        }
    }

    supp_g = supp_g_re;
    coeff_g = coeff_g_re;
    dj_g = dj_g_re;
    m_ineq = m_ineq_re;
    c_g = c_g_re;
    

    // equality constraints
    std::vector<Eigen::MatrixXi> supp_h_re;
    std::vector<Eigen::VectorXd> coeff_h_re;
    std::vector<int> dj_h_re;
    std::vector<int> c_h_re;
    int m_eq_re = 0;
    

    for (int i = 0; i < m_eq; ++i) {
        std::vector<int> nz_h;
        for (int row = 0; row < supp_h[i].rows(); ++row) {
            for (int col = 0; col < supp_h[i].cols(); ++col) {
                if (supp_h[i](row, col) != 0) {
                    nz_h.push_back(supp_h[i](row, col));
                }
            }
        }
        

        for (size_t j = 0; j < cI.size(); ++j) {
            bool is_subset = true;
            for (int var : nz_h) {
                if ((cI[j].array() == var).count() == 0) {
                    is_subset = false;
                    break;
                }
            }
            

            if (is_subset) {
                supp_h_re.push_back(supp_h[i]);
                coeff_h_re.push_back(coeff_h[i]);
                dj_h_re.push_back(dj_h[i]);
                c_h_re.push_back(j);
                m_eq_re++;
            }
            
        }
    }

    supp_h = supp_h_re;
    coeff_h = coeff_h_re;
    dj_h = dj_h_re;
    m_eq = m_eq_re;
    c_h = c_h_re;
}

//Record Monomials
void Polys::Record_monomials() {
    mon.resize(n_cI);
    mon_g.resize(m_ineq);
    mon_h.resize(m_eq);

    // 处理 mon
    for (int i = 0; i < n_cI; ++i) {
        mon[i] = build_union_vector(cI[i].size(), d, cI[i]);
    }

    // 处理 mon_g
    for (int i = 0; i < m_ineq; ++i) {
        mon_g[i] = build_union_vector(cI[c_g[i]].size(), d - dj_g[i], cI[c_g[i]]);
    }

    // 处理 mon_h
    for (int i = 0; i < m_eq; ++i) {
        mon_h[i] = build_union_vector(cI[c_h[i]].size(), 2 * d - dj_h[i], cI[c_h[i]]);
    }
}

//Construct C
void Polys::Construct_C() {
    // Combine supp_f, generate_2N_rpt(n, d), supp_g, and supp_h into one matrix
    Eigen::MatrixXi combine_all;

    // Combine supp_f
    combine_all = supp_f;

    // Combine generate_2N(n, d)
    if (d == 0) {
        Eigen::MatrixXi rpt = generate_2N(n, d);
        Eigen::MatrixXi temp(combine_all.rows() + rpt.rows(), combine_all.cols());
        temp << combine_all, rpt;
        combine_all = temp;
    }

    // Combine vertcat(supp_g{:}) (concatenate all supp_g matrices)
    for (const auto& g : supp_g) {
        Eigen::MatrixXi temp(combine_all.rows() + g.rows(), combine_all.cols());
        temp << combine_all, g;
        combine_all = temp;
    }

    // Combine vertcat(supp_h{:}) (concatenate all supp_h matrices)
    for (const auto& h : supp_h) {
        Eigen::MatrixXi temp(combine_all.rows() + h.rows(), combine_all.cols());
        temp << combine_all, h;
        combine_all = temp;
    }

    // Generate keys for each row
    std::vector<uint64_t> keys = get_key(combine_all, n);

    // Populate the unordered_map
    for (const auto& key : keys) {
        C[key] = 1; // Set all keys to 1
    }

}

// Term Sparsity
void Polys::Gen_tI(const std::string& ts_mode, const std::string& ts_mom_mode) {

    // 初始化 B_mom，B_ineq，B_eq
    std::vector<Eigen::MatrixXi> B_mom(n_cI), B_ineq(m_ineq), B_eq(m_eq);
    


    // Process support extension
    for (int i = 0; i < n_cI; ++i) {
        B_mom[i] = hs_support_extension_rpt(mon, C, Eigen::MatrixXi(), i, 0);
    }
    for (int i = 0; i < m_ineq; ++i) {
        B_ineq[i] = hs_support_extension_rpt(mon_g, C, supp_g[i], i, 1);
    }
    for (int i = 0; i < m_eq; ++i) {
        B_eq[i] = hs_support_extension_rpt(mon_h, C, supp_h[i], i, 2);
    }
    

    tI.resize(n_cI + m_ineq + m_eq);

    // Process term sparsity
    if (ts_mode == "MAX") {
        for (int itr = 0; itr < n_cI; ++itr) {
            B_mom[itr] = block_closure(B_mom[itr]);
            tI[itr] = find_blocks(B_mom[itr]);
        }
        for (int itr = 0; itr < m_ineq; ++itr) {
            B_ineq[itr] = block_closure(B_ineq[itr]);
            tI[n_cI + itr] = find_blocks(B_ineq[itr]);
        }
    } else if (ts_mode == "MD") {
        for (int itr = 0; itr < n_cI; ++itr) {
            tI[itr] = chordal_extension(B_mom[itr], 2);
            for (int i = 0; i < tI[itr].size(); ++i) {
                for (int j = 0; j < tI[itr][i].rows(); ++j) {
                    tI[itr][i](j) -= 1;
                }
            }
        }
        for (int itr = 0; itr < m_ineq; ++itr) {
            tI[n_cI + itr] = chordal_extension(B_ineq[itr], 2);
            for (int i = 0; i < tI[n_cI + itr].size(); ++i) {
                for (int j = 0; j < tI[n_cI + itr][i].rows(); ++j) {
                    tI[n_cI + itr][i](j) -= 1;
                }
            }            
        }
    } else if (ts_mode == "MF") {
        for (int itr = 0; itr < n_cI; ++itr) {
            tI[itr] = chordal_extension(B_mom[itr], 3);
            for (int i = 0; i < tI[itr].size(); ++i) {
                for (int j = 0; j < tI[itr][i].rows(); ++j) {
                    tI[itr][i](j) -= 1;
                }
            }
        }
        for (int itr = 0; itr < m_ineq; ++itr) {
            tI[n_cI + itr] = chordal_extension(B_ineq[itr], 3);
            for (int i = 0; i < tI[n_cI + itr].size(); ++i) {
                for (int j = 0; j < tI[n_cI + itr][i].rows(); ++j) {
                    tI[n_cI + itr][i](j) -= 1;
                }
            }            
        }
    } else if (ts_mode == "NON") {
        for (int itr = 0; itr < n_cI; ++itr) {
            Eigen::VectorXi full_block = Eigen::VectorXi::LinSpaced(B_mom[itr].rows(), 0, B_mom[itr].rows() - 1);
            tI[itr].push_back(full_block);
        }
        for (int itr = 0; itr < m_ineq; ++itr) {
            Eigen::VectorXi full_block = Eigen::VectorXi::LinSpaced(B_ineq[itr].rows(), 0, B_ineq[itr].rows() - 1);
            tI[n_cI + itr].push_back(full_block);
        }
    }
    

    // Add moment matrix
    if (ts_mom_mode == "USE") {
        for (int itr = 0; itr < n_cI; ++itr) {
            Eigen::VectorXi mom_block = Eigen::VectorXi::LinSpaced(cI[itr].size() + 1, 0, cI[itr].size());
            tI[itr].push_back(mom_block);
        }
    }

    // Compute tI_num
    tI_num.clear();
    for (int itr = 0; itr < n_cI + m_ineq; ++itr) {
        tI_num.push_back(tI[itr].size());
    }
    

    // Compute total number of sub-blocks
    n_tI = 0;
    for (int size : tI_num) {
        n_tI += size;
    }
    

    // Compute blocks size
    tI_size.resize(n_tI);
    for (int itr = 0; itr < n_tI; ++itr) {
        std::vector<int> itr_loca = find_loca_1(itr, tI_num);
        tI_size[itr] = tI[itr_loca[0]][itr_loca[1]].rows();
    }
    

    // Record B_eq
    for (int itr = 0; itr < m_eq; ++itr) {
        Eigen::VectorXi tI_tmp(B_eq[itr].rows());

        for (int row = 0; row < B_eq[itr].rows(); ++row) {
            tI_tmp(row) = B_eq[itr](row, 0);
        }
        tI[n_cI + m_ineq + itr].push_back(tI_tmp);
    }
    

}

//Moment Conversion
void Polys::Moment_Conversion(const std::string& ts_eq_mode) {

    int max_A = 0;
    for (int i1 = 0; i1 < n_cI; ++i1) {
        for (size_t i2 = 0; i2 < tI[i1].size(); ++i2) {
            max_A += tI[i1][i2].size() * tI[i1][i2].size();
        }
    }
    for (int i1 = 0; i1 < m_ineq; ++i1) {
        for (size_t i2 = 0; i2 < tI[n_cI + i1].size(); ++i2) {
            max_A += tI[n_cI + i1][i2].size() * tI[n_cI + i1][i2].size() * (coeff_g[i1].size() + 1);
        }
    }
    for (int i1 = 0; i1 < m_eq; ++i1) {
        max_A += tI[n_cI + m_ineq + i1][0].rows() * coeff_h[i1].size();
    }    

    // Initialize matrices
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(max_A, 5);
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(max_A, 4);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(max_A);

    std::unordered_map<uint64_t, std::vector<int>> mono_list;
    int index_A = 0;
    int itr_A = 0;
    int itr_C = 0;

    // step 1: moment
    for (int i1 = 0; i1 < n_cI; ++i1) {
        Eigen::MatrixXi mon_tmp = mon[i1];
        for (int i2 = 0; i2 < tI[i1].size(); ++i2) {
            for (int i3 = 0; i3 < tI[i1][i2].size(); ++i3) {
                for (int i4 = i3; i4 < tI[i1][i2].size(); ++i4) {
                    Eigen::MatrixXi combined(1, 2 * mon_tmp.cols());
                    combined << mon_tmp.row(tI[i1][i2](i3)), mon_tmp.row(tI[i1][i2](i4)); 
                    std::sort(combined.data(), combined.data() + combined.size());

                    std::vector<uint64_t> ij_key = get_key(combined, n);
                    bool is_in_list = std::all_of(ij_key.begin(), ij_key.end(),
                                                    [&](uint64_t key) { return mono_list.find(key) != mono_list.end(); });

                    if (is_in_list) {
                        auto ij_before = mono_list[ij_key[0]];
                        int ij_before_loca = find_loca_2({ij_before[0], ij_before[1]}, tI_num);
                        int ij_loca = find_loca_2({i1, i2}, tI_num);

                        if (ij_before[2] == ij_before[3]) {
                            A(itr_A, 0) = index_A;
                            A(itr_A, 1) = ij_before_loca;
                            A(itr_A, 2) = ij_before[2];
                            A(itr_A, 3) = ij_before[3];
                            A(itr_A, 4) = 1;
                        } else {
                            A(itr_A, 0) = index_A;
                            A(itr_A, 1) = ij_before_loca;
                            A(itr_A, 2) = ij_before[2];
                            A(itr_A, 3) = ij_before[3];
                            A(itr_A, 4) = 0.5;
                        }
                        ++itr_A;

                        if (i3 == i4) {
                            A(itr_A, 0) = index_A;
                            A(itr_A, 1) = ij_loca;
                            A(itr_A, 2) = i3;
                            A(itr_A, 3) = i4;
                            A(itr_A, 4) = -1;
                        } else {
                            A(itr_A, 0) = index_A;
                            A(itr_A, 1) = ij_loca;
                            A(itr_A, 2) = i3;
                            A(itr_A, 3) = i4;
                            A(itr_A, 4) = -0.5;
                        }
                        ++itr_A;
                        ++index_A;
                    } else {
                        mono_list[ij_key[0]] = {i1, i2, i3, i4};
                    }
                }
            }
        }
    }    

    
    // step 2: generate C
    std::vector<uint64_t> f_key = get_key(supp_f, n);
    for (int i = 0; i < f_key.size(); ++i) {
        auto f_before = mono_list[f_key[i]];
        int f_loca = find_loca_2({f_before[0], f_before[1]}, tI_num);

        if (f_before[2] == f_before[3]) {
            C(itr_C, 0) = f_loca;
            C(itr_C, 1) = f_before[2];
            C(itr_C, 2) = f_before[3];
            C(itr_C, 3) = coeff_f(i);
        } else {
            C(itr_C, 0) = f_loca;
            C(itr_C, 1) = f_before[2];
            C(itr_C, 2) = f_before[3];
            C(itr_C, 3) = 0.5 * coeff_f(i);
        }
        ++itr_C;
    }

    // step 3: inequality constraints
    for (int i1 = 0; i1 < m_ineq; ++i1) {
        Eigen::MatrixXi mon_tmp = mon_g[i1];
        Eigen::MatrixXi supp_rpt_g = supp_g[i1];
        for (int i2 = 0; i2 < tI[n_cI + i1].size(); ++i2) {
            for (int i3 = 0; i3 < tI[n_cI + i1][i2].size(); ++i3) {
                for (int i4 = i3; i4 < tI[n_cI + i1][i2].size(); ++i4) {
                    Eigen::MatrixXi combined(supp_rpt_g.rows(), supp_rpt_g.cols() + mon_tmp.cols() * 2);
                    combined.block(0, 0, supp_rpt_g.rows(), mon_tmp.cols()) = mon_tmp.row(tI[n_cI + i1][i2](i3)).replicate(supp_rpt_g.rows(), 1);
                    combined.block(0, mon_tmp.cols(), supp_rpt_g.rows(), mon_tmp.cols()) = mon_tmp.row(tI[n_cI + i1][i2](i4)).replicate(supp_rpt_g.rows(), 1);
                    combined.block(0, 2 * mon_tmp.cols(), supp_rpt_g.rows(), supp_rpt_g.cols()) = supp_rpt_g;
                    for (int k = 0; k < combined.rows(); ++k) {
                        Eigen::VectorXi row = combined.row(k);
                        std::sort(row.data(), row.data() + row.size());
                        combined.row(k) = row;
                    }

                    std::vector<uint64_t> ij_key = get_key(combined, n);
                    bool is_in_list = std::all_of(ij_key.begin(), ij_key.end(),
                                                  [&](uint64_t key) { return mono_list.find(key) != mono_list.end(); });

                    if (is_in_list) {
                        int ij_loca = find_loca_2({n_cI + i1, i2}, tI_num);

                        if (i3 == i4) {
                            A(itr_A, 0) = index_A;
                            A(itr_A, 1) = ij_loca;
                            A(itr_A, 2) = i3;
                            A(itr_A, 3) = i4;
                            A(itr_A, 4) = -1;
                        } else {
                            A(itr_A, 0) = index_A;
                            A(itr_A, 1) = ij_loca;
                            A(itr_A, 2) = i3;
                            A(itr_A, 3) = i4;
                            A(itr_A, 4) = -0.5;
                        }
                        ++itr_A;

                        for (int i5 = 0; i5 < ij_key.size(); ++i5) {
                            auto ij_before = mono_list[ij_key[i5]];
                            int ij_before_loca = find_loca_2({ij_before[0], ij_before[1]}, tI_num);

                            if (ij_before[2] == ij_before[3]) {
                                A(itr_A, 0) = index_A;
                                A(itr_A, 1) = ij_before_loca;
                                A(itr_A, 2) = ij_before[2];
                                A(itr_A, 3) = ij_before[3];
                                A(itr_A, 4) = coeff_g[i1](i5);
                            } else {
                                A(itr_A, 0) = index_A;
                                A(itr_A, 1) = ij_before_loca;
                                A(itr_A, 2) = ij_before[2];
                                A(itr_A, 3) = ij_before[3];
                                A(itr_A, 4) = 0.5 * coeff_g[i1](i5);
                            }
                            ++itr_A;
                        }
                        ++index_A;
                    }
                }
            }
        }
    }

    // step 4: equality constraints
    for (int i1 = 0; i1 < m_eq; ++i1) {
        Eigen::MatrixXi mon_tmp = mon_h[i1];
        Eigen::MatrixXi supp_rpt_h = supp_h[i1];
        for (int i2 = 0; i2 < mon_tmp.rows(); ++i2) {
            // check ts_eq_mode
            if (ts_mode == "USE" && ts_eq_mode == "USE" && tI[n_cI + m_ineq + i1][0](i2) == 0) {
                continue;
            }

            Eigen::MatrixXi combined(supp_rpt_h.rows(), supp_rpt_h.cols() + mon_tmp.cols());
            combined.block(0, 0, supp_rpt_h.rows(), mon_tmp.cols()) = mon_tmp.row(i2).replicate(supp_rpt_h.rows(), 1);
            combined.block(0, mon_tmp.cols(), supp_rpt_h.rows(), supp_rpt_h.cols()) = supp_rpt_h;

            for (int k = 0; k < combined.rows(); ++k) {
                Eigen::VectorXi row = combined.row(k);
                std::sort(row.data(), row.data() + row.size());
                combined.row(k) = row;
            }

            std::vector<uint64_t> i_key = get_key(combined, n);
            bool is_in_list = std::all_of(i_key.begin(), i_key.end(),
                                        [&](uint64_t key) { return mono_list.find(key) != mono_list.end(); });

            if (is_in_list) {
                for (int i5 = 0; i5 < i_key.size(); ++i5) {
                    auto i_before = mono_list[i_key[i5]];
                    int i_before_loca = find_loca_2({i_before[0], i_before[1]}, tI_num);

                    if (i_before[2] == i_before[3]) {
                        A(itr_A, 0) = index_A;
                        A(itr_A, 1) = i_before_loca;
                        A(itr_A, 2) = i_before[2];
                        A(itr_A, 3) = i_before[3];
                        A(itr_A, 4) = coeff_h[i1](i5);
                    } else {
                        A(itr_A, 0) = index_A;
                        A(itr_A, 1) = i_before_loca;
                        A(itr_A, 2) = i_before[2];
                        A(itr_A, 3) = i_before[3];
                        A(itr_A, 4) = 0.5 * coeff_h[i1](i5);
                    }
                    ++itr_A;
                }
                ++index_A;
            }
        }
    }

    // step 5: y_00 = 1
    for (int i = 0; i < n_cI; ++i) {
        if (std::find(cI[i].data(), cI[i].data() + cI[i].size(), 1) != cI[i].data() + cI[i].size()) {
            int loca = find_loca_2({i, 0}, tI_num);
            A(itr_A, 0) = index_A;
            A(itr_A, 1) = loca;
            A(itr_A, 2) = 0;
            A(itr_A, 3) = 0;
            A(itr_A, 4) = 1;
            b(index_A) = 1;
            ++itr_A;
            ++index_A;
        }
    }

    // step 6: delete redundant
    A.conservativeResize(itr_A, Eigen::NoChange);
    b.conservativeResize(index_A);
    C.conservativeResize(itr_C, Eigen::NoChange);

    A_moment = A;
    b_moment = b;
    C_moment = C;

}

// SOS Conversion
void Polys::SOS_Conversion(const std::string& ts_eq_mode) {
    // Step 0: Initialize matrices
    int max_A = 0;
    for (int i1 = 0; i1 < n_cI; ++i1) {
        for (size_t i2 = 0; i2 < tI[i1].size(); ++i2) {
            max_A += tI[i1][i2].size() * tI[i1][i2].size();
        }
    }
    for (int i1 = 0; i1 < m_ineq; ++i1) {
        for (size_t i2 = 0; i2 < tI[n_cI + i1].size(); ++i2) {
            max_A += tI[n_cI + i1][i2].size() * tI[n_cI + i1][i2].size() * (coeff_g[i1].size() + 1);
        }
    }
    for (int i1 = 0; i1 < m_eq; ++i1) {
        max_A += tI[n_cI + m_ineq + i1][0].rows() * coeff_h[i1].size();
    }
    

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(max_A, 5);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(max_A);
    Eigen::MatrixXd prob_a = Eigen::MatrixXd::Zero(max_A, 3);

    std::unordered_map<uint64_t, double> index_f;
    std::unordered_map<uint64_t, int> alpha_list;

    int index_A = 0;  // Constraint index
    int itr_A = 0;
    int itr_a = 0;
    int free_index = 0;  // First free variable (rho)

    // Step 1: Initialize index_f
    std::vector<uint64_t> f_key = get_key(supp_f, n);
    for (size_t i = 0; i < f_key.size(); ++i) {
        index_f[f_key[i]] = coeff_f(i);
    }
    

    // Step 2: Alpha = 0
    alpha_list[0] = index_A;
    if ((supp_f.row(0).array() == 0).all()) {
        b(index_A) = coeff_f(0);
    } else {
        b(index_A) = 0;
    }
    ++index_A;

    // Step 2: Alpha in moment constraints
    for (int i1 = 0; i1 < n_cI; ++i1) {
        Eigen::MatrixXi mon_tmp = mon[i1];
        for (int i2 = 0; i2 < tI[i1].size(); ++i2) {
            for (int i3 = 0; i3 < tI[i1][i2].size(); ++i3) {
                for (int i4 = i3; i4 < tI[i1][i2].size(); ++i4) {
                    Eigen::MatrixXi combined(1, 2 * mon_tmp.cols());
                    combined << mon_tmp.row(tI[i1][i2](i3)), mon_tmp.row(tI[i1][i2](i4)); 
                    std::sort(combined.data(), combined.data() + combined.size());
                    std::vector<uint64_t> ij_key = get_key(combined, n);
                    int ij_loca = find_loca_2({i1, i2}, tI_num);

                    for (size_t i5 = 0; i5 < ij_key.size(); ++i5) {
                        if (alpha_list.find(ij_key[i5]) == alpha_list.end()) {
                            A(itr_A, 0) = index_A;
                            A(itr_A, 1) = ij_loca;
                            A(itr_A, 2) = i3;
                            A(itr_A, 3) = i4;
                            A(itr_A, 4) = 1;
                            ++itr_A;

                            if (index_f.find(ij_key[i5]) != index_f.end()) {
                                b(index_A) = index_f[ij_key[i5]];
                            }
                            alpha_list[ij_key[i5]] = index_A;
                            ++index_A;
                        } else {
                            int index_A_before = alpha_list[ij_key[i5]];
                            A(itr_A, 0) = index_A_before;
                            A(itr_A, 1) = ij_loca;
                            A(itr_A, 2) = i3;
                            A(itr_A, 3) = i4;
                            A(itr_A, 4) = 1;
                            ++itr_A;
                        }
                    }
                }
            }
        }
    }
    std::cout << index_A << "  " << itr_A << std::endl;
    
    // Step 3: Alpha in inequality constraints
    for (int i1 = 0; i1 < m_ineq; ++i1) {
        Eigen::MatrixXi mon_tmp = mon_g[i1];
        Eigen::MatrixXi supp_rpt_g = supp_g[i1];
        for (int i2 = 0; i2 < tI[n_cI + i1].size(); ++i2) {
            for (int i3 = 0; i3 < tI[n_cI + i1][i2].size(); ++i3) {
                for (int i4 = i3; i4 < tI[n_cI + i1][i2].size(); ++i4) {

                    Eigen::MatrixXi combined(supp_rpt_g.rows(), supp_rpt_g.cols() + mon_tmp.cols() * 2);
                    combined.block(0, 0, supp_rpt_g.rows(), mon_tmp.cols()) = mon_tmp.row(tI[n_cI + i1][i2](i3)).replicate(supp_rpt_g.rows(), 1);
                    combined.block(0, mon_tmp.cols(), supp_rpt_g.rows(), mon_tmp.cols()) = mon_tmp.row(tI[n_cI + i1][i2](i4)).replicate(supp_rpt_g.rows(), 1);
                    combined.block(0, 2 * mon_tmp.cols(), supp_rpt_g.rows(), supp_rpt_g.cols()) = supp_rpt_g;

                    for (int k = 0; k < combined.rows(); ++k) {
                        Eigen::VectorXi row = combined.row(k);
                        std::sort(row.data(), row.data() + row.size());
                        combined.row(k) = row;
                    }
                    // printdebug(combined);
                    std::vector<uint64_t> ij_key = get_key(combined, n);

                    int ij_loca = find_loca_2({n_cI + i1, i2}, tI_num);

                    for (size_t i5 = 0; i5 < ij_key.size(); ++i5) {
                        if (alpha_list.find(ij_key[i5]) == alpha_list.end()) {
                            A(itr_A, 0) = index_A;
                            A(itr_A, 1) = ij_loca;
                            A(itr_A, 2) = i3;
                            A(itr_A, 3) = i4;
                            A(itr_A, 4) = coeff_g[i1](i5);
                            ++itr_A;

                            if (index_f.find(ij_key[i5]) != index_f.end()) {
                                b(index_A) = index_f[ij_key[i5]];
                            }
                            alpha_list[ij_key[i5]] = index_A;
                            ++index_A;
                        } else {
                            int index_A_before = alpha_list[ij_key[i5]];
                            A(itr_A, 0) = index_A_before;
                            A(itr_A, 1) = ij_loca;
                            A(itr_A, 2) = i3;
                            A(itr_A, 3) = i4;
                            A(itr_A, 4) = coeff_g[i1](i5);
                            ++itr_A;
                        }
                    }
                }
            }
        }
    }
    std::cout << index_A << "  " << itr_A << std::endl;

    // Step 4: Equality constraints
    for (int i1 = 0; i1 < m_eq; ++i1) {
        Eigen::MatrixXi mon_tmp = mon_h[i1];
        Eigen::MatrixXi supp_rpt_h = supp_h[i1];
        for (int i2 = 0; i2 < mon_tmp.rows(); ++i2) {
            // check ts_eq_mode
            if (ts_mode == "USE" && ts_eq_mode == "USE" && tI[n_cI + m_ineq + i1][0](i2) == 0) {
                continue;
            }

            ++free_index;

            Eigen::MatrixXi combined(supp_rpt_h.rows(), supp_rpt_h.cols() + mon_tmp.cols());
            combined.block(0, 0, supp_rpt_h.rows(), mon_tmp.cols()) = mon_tmp.row(i2).replicate(supp_rpt_h.rows(), 1);
            combined.block(0, mon_tmp.cols(), supp_rpt_h.rows(), supp_rpt_h.cols()) = supp_rpt_h;

            for (int k = 0; k < combined.rows(); ++k) {
                Eigen::VectorXi row = combined.row(k);
                std::sort(row.data(), row.data() + row.size());
                combined.row(k) = row;
            }

            std::vector<uint64_t> i_key = get_key(combined, n);

            for (size_t i5 = 0; i5 < i_key.size(); ++i5) {
                if (alpha_list.find(i_key[i5]) == alpha_list.end()) {
                    prob_a(itr_a, 0) = index_A;
                    prob_a(itr_a, 1) = free_index;
                    prob_a(itr_a, 2) = coeff_h[i1](i5);
                    ++itr_a;

                    alpha_list[i_key[i5]] = index_A;
                    ++index_A;
                } else {
                    int index_A_before = alpha_list[i_key[i5]];
                    prob_a(itr_a, 0) = index_A_before;
                    prob_a(itr_a, 1) = free_index;
                    prob_a(itr_a, 2) = coeff_h[i1](i5);
                    ++itr_a;
                }
            }
        }
    }
    std::cout << index_A << "  " << itr_A << "  " << itr_a << "  "<< free_index << std::endl;

    // Step 5: Remove redundant
    A.conservativeResize(itr_A, Eigen::NoChange);
    b.conservativeResize(index_A);
    prob_a.conservativeResize(itr_a + 1, Eigen::NoChange);
    prob_a.row(itr_a) << 0, 0, 1;

    A_sos = A;
    b_sos = b;
    a_sos = prob_a;
    c_sos = Eigen::VectorXi::Zero(free_index + 1); 
    c_sos(0) = 1; 
}


// change_idx: index of changed constraints; change_coeff: changed coeff_g and coeff_h
void Polys::CSTSS_Real(const std::vector<int>& change_idx_g, const std::vector<int>& change_idx_h,
                        const std::vector<Eigen::VectorXd>& change_coeff_g, const std::vector<Eigen::VectorXd>& change_coeff_h, const int if_first) {
    if (if_first) {
        this->Gen_cI(cs_mode);
        this->Assign_Constraints();
        this->Record_monomials();
        this->Construct_C();
        this->Gen_tI(ts_mode, ts_mom_mode);
        this->SOS_Conversion_Real(ts_eq_mode);
    } else {
        if (!change_idx_g.empty()) {
            for (int i = 0; i < idx_ineq.size(); ++i) {
                auto idx = find(change_idx_g.begin(), change_idx_g.end(), idx_ineq[i][1]); // the second element is i1
                if (idx != change_idx_g.end()) {
                    A_sos(idx_ineq[i][0], 4) = change_coeff_g[idx_ineq[i][1]](idx_ineq[i][2]);
                }                
            }
        }
        if (!change_idx_h.empty()) {
            for (int i = 0; i < idx_eq.size(); ++i) {
                auto idx = find(change_idx_h.begin(), change_idx_h.end(), idx_eq[i][1]); // the second element is i1
                if (idx != change_idx_h.end()) {
                    a_sos(idx_ineq[i][0], 2) = change_coeff_h[idx_eq[i][1]](idx_eq[i][2]);
                }                
            }            
        }
    }
}

void Polys::SOS_Conversion_Real(const std::string& ts_eq_mode) {
    // Step 0: Initialize matrices
    int max_A = 0;
    for (int i1 = 0; i1 < n_cI; ++i1) {
        for (size_t i2 = 0; i2 < tI[i1].size(); ++i2) {
            max_A += tI[i1][i2].size() * tI[i1][i2].size();
        }
    }
    
    for (int i1 = 0; i1 < m_ineq; ++i1) {
        for (size_t i2 = 0; i2 < tI[n_cI + i1].size(); ++i2) {
            max_A += tI[n_cI + i1][i2].size() * tI[n_cI + i1][i2].size() * (coeff_g[i1].size() + 1);
        }
    }
    
    for (int i1 = 0; i1 < m_eq; ++i1) {
        max_A += tI[n_cI + m_ineq + i1][0].rows() * coeff_h[i1].size();
    }
    

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(max_A, 5);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(max_A);
    Eigen::MatrixXd prob_a = Eigen::MatrixXd::Zero(max_A, 3);

    std::unordered_map<uint64_t, double> index_f;
    std::unordered_map<uint64_t, int> alpha_list;

    int index_A = 0;  // Constraint index
    int itr_A = 0;
    int itr_a = 0;
    int free_index = 0;  // First free variable (rho)

    // Step 1: Initialize index_f
    std::vector<uint64_t> f_key = get_key(supp_f, n);
    for (size_t i = 0; i < f_key.size(); ++i) {
        index_f[f_key[i]] = coeff_f(i);
    }
    
    

    // Step 2: Alpha = 0
    alpha_list[0] = index_A;
    if ((supp_f.row(0).array() == 0).all()) {
        b(index_A) = coeff_f(0);
    } else {
        b(index_A) = 0;
    }
    ++index_A;
    

    // Step 2: Alpha in moment constraints
    for (int i1 = 0; i1 < n_cI; ++i1) {
        Eigen::MatrixXi mon_tmp = mon[i1];
        for (int i2 = 0; i2 < tI[i1].size(); ++i2) {
            for (int i3 = 0; i3 < tI[i1][i2].size(); ++i3) {
                for (int i4 = i3; i4 < tI[i1][i2].size(); ++i4) {
                    Eigen::MatrixXi combined(1, 2 * mon_tmp.cols());
                    combined << mon_tmp.row(tI[i1][i2](i3)), mon_tmp.row(tI[i1][i2](i4)); 
                    std::sort(combined.data(), combined.data() + combined.size());
                    std::vector<uint64_t> ij_key = get_key(combined, n);
                    int ij_loca = find_loca_2({i1, i2}, tI_num);

                    for (int i5 = 0; i5 < ij_key.size(); ++i5) {
                        if (alpha_list.find(ij_key[i5]) == alpha_list.end()) {
                            A(itr_A, 0) = index_A;
                            A(itr_A, 1) = ij_loca;
                            A(itr_A, 2) = i3;
                            A(itr_A, 3) = i4;
                            A(itr_A, 4) = 1;
                            ++itr_A;

                            if (index_f.find(ij_key[i5]) != index_f.end()) {
                                b(index_A) = index_f[ij_key[i5]];
                            }
                            alpha_list[ij_key[i5]] = index_A;
                            ++index_A;
                        } else {
                            int index_A_before = alpha_list[ij_key[i5]];
                            A(itr_A, 0) = index_A_before;
                            A(itr_A, 1) = ij_loca;
                            A(itr_A, 2) = i3;
                            A(itr_A, 3) = i4;
                            A(itr_A, 4) = 1;
                            ++itr_A;
                        }
                    }
                }
            }
        }
    }
    std::cout << index_A << "  " << itr_A << std::endl;
    
    
    // Step 3: Alpha in inequality constraints
    for (int i1 = 0; i1 < m_ineq; ++i1) {
        Eigen::MatrixXi mon_tmp = mon_g[i1];
        Eigen::MatrixXi supp_rpt_g = supp_g[i1];
        for (int i2 = 0; i2 < tI[n_cI + i1].size(); ++i2) {
            for (int i3 = 0; i3 < tI[n_cI + i1][i2].size(); ++i3) {
                for (int i4 = i3; i4 < tI[n_cI + i1][i2].size(); ++i4) {

                    Eigen::MatrixXi combined(supp_rpt_g.rows(), supp_rpt_g.cols() + mon_tmp.cols() * 2);
                    combined.block(0, 0, supp_rpt_g.rows(), mon_tmp.cols()) = mon_tmp.row(tI[n_cI + i1][i2](i3)).replicate(supp_rpt_g.rows(), 1);
                    combined.block(0, mon_tmp.cols(), supp_rpt_g.rows(), mon_tmp.cols()) = mon_tmp.row(tI[n_cI + i1][i2](i4)).replicate(supp_rpt_g.rows(), 1);
                    combined.block(0, 2 * mon_tmp.cols(), supp_rpt_g.rows(), supp_rpt_g.cols()) = supp_rpt_g;

                    for (int k = 0; k < combined.rows(); ++k) {
                        Eigen::VectorXi row = combined.row(k);
                        std::sort(row.data(), row.data() + row.size());
                        combined.row(k) = row;
                    }
                    // printdebug(combined);
                    std::vector<uint64_t> ij_key = get_key(combined, n);

                    int ij_loca = find_loca_2({n_cI + i1, i2}, tI_num);

                    for (int i5 = 0; i5 < ij_key.size(); ++i5) {
                        if (alpha_list.find(ij_key[i5]) == alpha_list.end()) {
                            A(itr_A, 0) = index_A;
                            A(itr_A, 1) = ij_loca;
                            A(itr_A, 2) = i3;
                            A(itr_A, 3) = i4;
                            A(itr_A, 4) = coeff_g[i1](i5);
                            ++itr_A;

                            // For Real 
                            idx_ineq.push_back(std::vector<int>{itr_A - 1, i1, i5});

                            if (index_f.find(ij_key[i5]) != index_f.end()) {
                                b(index_A) = index_f[ij_key[i5]];
                            }
                            alpha_list[ij_key[i5]] = index_A;
                            ++index_A;
                        } else {
                            int index_A_before = alpha_list[ij_key[i5]];
                            A(itr_A, 0) = index_A_before;
                            A(itr_A, 1) = ij_loca;
                            A(itr_A, 2) = i3;
                            A(itr_A, 3) = i4;
                            A(itr_A, 4) = coeff_g[i1](i5);
                            ++itr_A;

                            // For Real 
                            idx_ineq.push_back(std::vector<int>{itr_A - 1, i1, i5});                            
                        }
                    }
                }
            }
        }
    }
    std::cout << index_A << "  " << itr_A << std::endl;

    // Step 4: Equality constraints
    for (int i1 = 0; i1 < m_eq; ++i1) {
        Eigen::MatrixXi mon_tmp = mon_h[i1];
        Eigen::MatrixXi supp_rpt_h = supp_h[i1];
        for (int i2 = 0; i2 < mon_tmp.rows(); ++i2) {
            // check ts_eq_mode
            if (ts_mode == "USE" && ts_eq_mode == "USE" && tI[n_cI + m_ineq + i1][0](i2) == 0) {
                continue;
            }

            ++free_index;

            Eigen::MatrixXi combined(supp_rpt_h.rows(), supp_rpt_h.cols() + mon_tmp.cols());
            combined.block(0, 0, supp_rpt_h.rows(), mon_tmp.cols()) = mon_tmp.row(i2).replicate(supp_rpt_h.rows(), 1);
            combined.block(0, mon_tmp.cols(), supp_rpt_h.rows(), supp_rpt_h.cols()) = supp_rpt_h;

            for (int k = 0; k < combined.rows(); ++k) {
                Eigen::VectorXi row = combined.row(k);
                std::sort(row.data(), row.data() + row.size());
                combined.row(k) = row;
            }

            std::vector<uint64_t> i_key = get_key(combined, n);

            for (int i5 = 0; i5 < i_key.size(); ++i5) {
                if (alpha_list.find(i_key[i5]) == alpha_list.end()) {
                    prob_a(itr_a, 0) = index_A;
                    prob_a(itr_a, 1) = free_index;
                    prob_a(itr_a, 2) = coeff_h[i1](i5);
                    ++itr_a;
                    
                    // For Real
                    idx_eq.push_back(std::vector<int>{itr_a - 1, i1, i5});

                    alpha_list[i_key[i5]] = index_A;
                    ++index_A;
                } else {
                    int index_A_before = alpha_list[i_key[i5]];
                    prob_a(itr_a, 0) = index_A_before;
                    prob_a(itr_a, 1) = free_index;
                    prob_a(itr_a, 2) = coeff_h[i1](i5);
                    ++itr_a;

                    // For Real
                    idx_eq.push_back(std::vector<int>{itr_a - 1, i1, i5});                    
                }
            }
        }
    }
    std::cout << index_A << "  " << itr_A << "  " << itr_a << "  "<< free_index << std::endl;

    // Step 5: Remove redundant
    A.conservativeResize(itr_A, Eigen::NoChange);
    b.conservativeResize(index_A);
    prob_a.conservativeResize(itr_a + 1, Eigen::NoChange);
    prob_a.row(itr_a) << 0, 0, 1;

    A_sos = A;
    b_sos = b;
    a_sos = prob_a;
    c_sos = Eigen::VectorXi::Zero(free_index + 1); 
    c_sos(0) = 1; 
}
























// Auxiliary Function

Eigen::MatrixXi Polys::fill_edges(Eigen::MatrixXi& G, const std::vector<int>& neighbors) {
    for (size_t i = 0; i < neighbors.size(); ++i) {
        for (size_t j = i + 1; j < neighbors.size(); ++j) {
            G(neighbors[i], neighbors[j]) = 1;
            G(neighbors[j], neighbors[i]) = 1;
        }
    }
    return G;
}

Eigen::MatrixXi Polys::maximal_chordal_extension(const Eigen::MatrixXi& G) {
    return block_closure(G);
}

std::vector<Eigen::VectorXi> Polys::find_max_cliques(const Eigen::MatrixXi& G, const std::vector<int>& order, int mode) {
    int n = G.rows();
    std::vector<Eigen::VectorXi> cliques;

    if (mode == 1) {
        // for maximal chordal extension
        find_blocks(G);
    } else if (mode == 2) {
        // Step 1: 构造每个顶点的临时 clique
        std::vector<Eigen::VectorXi> tmp_cliques(n);
        for (int i = 0; i < n; ++i) {
            int v = i;
            std::vector<int> clique;

            // 当前顶点及其满足条件的邻居加入 clique
            clique.push_back(v + 1);  // MATLAB 索引从 1 开始
            for (int j = 0; j < n; ++j) {
                if (G(v, j) == 1 && order[j] > order[v]) {
                    clique.push_back(j + 1);
                }
            }

            // 排序并存储为 Eigen::VectorXi
            std::sort(clique.begin(), clique.end());
            Eigen::VectorXi clique_vec = Eigen::VectorXi::Map(clique.data(), clique.size());
            tmp_cliques[i] = clique_vec;
        }

        // Step 2: 按 clique 大小降序排序
        std::vector<int> clique_sizes(n);
        for (int i = 0; i < n; ++i) {
            clique_sizes[i] = tmp_cliques[i].size();
        }
        std::vector<int> sorted_indices(n);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(), [&](int a, int b) {
            return clique_sizes[a] > clique_sizes[b];
        });

        std::vector<Eigen::VectorXi> sorted_cliques(n);
        for (int i = 0; i < n; ++i) {
            sorted_cliques[i] = tmp_cliques[sorted_indices[i]];
        }

        // Step 3: 构造 maximal_cliques
        std::vector<Eigen::VectorXi> maximal_cliques;
        maximal_cliques.push_back(sorted_cliques[0]);  // 第一个 clique 一定是极大 clique

        for (int i = 1; i < n; ++i) {
            const Eigen::VectorXi& current_clique = sorted_cliques[i];
            bool is_contained = false;

            // 检查是否被已有的 maximal_cliques 包含
            for (const auto& maximal_clique : maximal_cliques) {
                if (std::includes(maximal_clique.data(), maximal_clique.data() + maximal_clique.size(),
                                  current_clique.data(), current_clique.data() + current_clique.size())) {
                    is_contained = true;
                    break;
                }
            }

            // 如果未被包含，加入 maximal_cliques
            if (!is_contained) {
                maximal_cliques.push_back(current_clique);
            }
        }

        cliques = maximal_cliques;
    }

    return cliques;
}

Eigen::MatrixXi Polys::greedy_chordal_extension(const Eigen::MatrixXi& G, std::vector<int>& order) {
    int n = G.rows();
    Eigen::MatrixXi G_chordal = G; // 初始化 chordal 图
    G_chordal.diagonal().setZero(); // 确保对角线为 0
    Eigen::MatrixXi H = G; // 工作图 H
    H.diagonal().setZero(); // 确保对角线为 0
    order.resize(n, 0); // 存储每个点的消除顺序

    Eigen::VectorXi degrees = H.rowwise().sum();

    for (int i = 0; i < n; ++i) {
        int v = 0;
        for (int j = 0; j < n; ++j) {
            if (order[j] == 0 && degrees(j) < degrees(v)) {
                v = j;
            }
        }

        order[v] = i + 1;    

        std::vector<int> neighbors_v;
        for (int j = 0; j < n; ++j) {
            if (H(v, j) == 1) {
                neighbors_v.push_back(j);
            }
        }

        for (int k = 0; k < neighbors_v.size(); ++k) {
            H(v, neighbors_v[k]) = 0;
            H(neighbors_v[k], v) = 0;
            degrees(neighbors_v[k]) -= 1;
            for (int t = k + 1; t < neighbors_v.size(); ++t) {
                if (H(neighbors_v[k], neighbors_v[t]) == 0) {
                    H(neighbors_v[k], neighbors_v[t]) = 1; degrees(neighbors_v[k]) += 1;
                    H(neighbors_v[t], neighbors_v[k]) = 1; degrees(neighbors_v[t]) += 1;
                }
            }
        }
        for (int k = 0; k < neighbors_v.size(); ++k) {
            for (int t = k + 1; t < neighbors_v.size(); ++t) {
                G_chordal(neighbors_v[k], neighbors_v[t]) = 1;
                G_chordal(neighbors_v[t], neighbors_v[k]) = 1;
            }
        }

        // update degrees
        degrees(v) = std::numeric_limits<int>::max();
    }

    return G_chordal;
}

// Eigen::MatrixXi Polys::minimal_fill_chordal_extension(const Eigen::MatrixXi& G, std::vector<int>& order) {
//     int n = G.rows();
//     Eigen::MatrixXi G_chordal = G;  
//     G_chordal.diagonal().setZero();
//     Eigen::MatrixXi H = G;       
//     H.diagonal().setZero();      
//     order.resize(n, 0);           

//     std::vector<int> fill_counts(n, 0);

//     for (int v = 0; v < n; ++v) {
//         std::vector<int> neighbors_v;
//         for (int j = 0; j < n; ++j) {
//             if (H(v, j) == 1) {
//                 neighbors_v.push_back(j);
//             }
//         }

//         int fill_count = 0;
//         int degree = neighbors_v.size();
//         for (int j = 0; j < degree; ++j) {
//             for (int k = j + 1; k < degree; ++k) {
//                 if (H(neighbors_v[j], neighbors_v[k]) == 0) {
//                     ++fill_count;
//                 }
//             }
//         }
//         fill_counts[v] = fill_count;
//     }    

//     for (int i = 0; i < n; ++i) {

//         int v = std::min_element(fill_counts.begin(), fill_counts.end()) - fill_counts.begin();
//         order[v] = i + 1; 
//         fill_counts[v] = std::numeric_limits<int>::max();

//         std::vector<int> neighbors_v;
//         for (int j = 0; j < n; ++j) {
//             if (H(v, j) == 1) {
//                 neighbors_v.push_back(j);
//             }
//         }

//         auto H_tmp = H;
//         for (int j = 0; j < neighbors_v.size(); ++j) {
//             H_tmp(v, neighbors_v[j]) = 0;
//             H_tmp(neighbors_v[j], v) = 0;
//             for (int k = 0; k < neighbors_v.size(); ++k) {
//                 if (H(neighbors_v[j], neighbors_v[k]) == 0) {
//                     H_tmp(neighbors_v[j], neighbors_v[k]) = 1; 
//                 } else {
//                     for (int t = 0; t < neighbors_v.size(); ++t) {
//                         if (H(neighbors_v[j], neighbors_v[t]) == 1 && H(neighbors_v[t], neighbors_v[k]) == 0 && t != j && t != k) {
//                             fill_counts[j] -= 1;
//                         }
//                     }
//                 }
//             }
//         }
//         H = H_tmp;

//         for (int k = 0; k < neighbors_v.size(); ++k) {
//             for (int t = k + 1; t < neighbors_v.size(); ++t) {
//                 G_chordal(neighbors_v[k], neighbors_v[t]) = 1;
//                 G_chordal(neighbors_v[t], neighbors_v[k]) = 1;
//             }
//         }        
//     }

//     return G_chordal;
// }

Eigen::MatrixXi Polys::minimal_fill_chordal_extension(const Eigen::MatrixXi& G, std::vector<int>& order) {
    int n = G.rows();
    Eigen::MatrixXi G_chordal = G; // 输出的弦图扩展矩阵
    G_chordal.diagonal().setZero();
    Eigen::MatrixXi H = G;
    H.diagonal().setZero();
    order.resize(n, 0);           // 消除顺序

    // 邻接表表示，用于快速操作图结构
    std::vector<std::set<int>> adjList(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (H(i, j) == 1) {
                adjList[i].insert(j);
            }
        }
    }

    for (int i = 1; i <= n; ++i) {
        // 记录填充边的数量
        std::vector<int> fillCounts(n, std::numeric_limits<int>::max());

        for (int v = 0; v < n; ++v) {
            if (order[v] > 0) continue; // 跳过已被选过的顶点

            const auto& neighbors = adjList[v];
            int fillCount = 0;

            // 计算填充边数（缺失边数）
            for (auto it1 = neighbors.begin(); it1 != neighbors.end(); ++it1) {
                auto it2 = it1;
                ++it2;
                for (; it2 != neighbors.end(); ++it2) {
                    if (adjList[*it1].count(*it2) == 0) {
                        fillCount++;
                    }
                }
            }
            fillCounts[v] = fillCount;
        }

        // 选择填充边数最小的顶点
        int v_idx = std::min_element(fillCounts.begin(), fillCounts.end()) - fillCounts.begin();
        order[v_idx] = i;

        // 更新图 H 和 G_chordal
        const auto& neighbors = adjList[v_idx];
        for (auto it1 = neighbors.begin(); it1 != neighbors.end(); ++it1) {
            auto it2 = it1;
            ++it2;
            for (; it2 != neighbors.end(); ++it2) {
                // 在 H 和 G_chordal 中填充边
                if (adjList[*it1].count(*it2) == 0) {
                    adjList[*it1].insert(*it2);
                    adjList[*it2].insert(*it1);
                    G_chordal(*it1, *it2) = 1;
                    G_chordal(*it2, *it1) = 1;
                }
            }
        }

        // 从邻接表中删除顶点 v_idx
        for (auto neighbor : neighbors) {
            adjList[neighbor].erase(v_idx);
        }
        adjList[v_idx].clear();
    }

    return G_chordal;
}

// Csp Construction
Eigen::MatrixXi Polys::csp_construct() {
    Eigen::MatrixXi G = Eigen::MatrixXi::Zero(n, n);

    // 缓存 supp_f 的非零索引
    std::vector<std::vector<int>> supp_f_indices(supp_f.rows());
    for (int i = 0; i < supp_f.rows(); ++i) {
        for (int j = 0; j < supp_f.cols(); ++j) {
            if (supp_f(i, j) > 0) {
                supp_f_indices[i].push_back(supp_f(i, j) - 1);
            }
        }
    }

    // 构造 G 矩阵
    for (const auto& indices : supp_f_indices) {
        for (size_t j = 0; j < indices.size(); ++j) {
            for (size_t k = j + 1; k < indices.size(); ++k) {
                G(indices[j], indices[k]) = 1;
                G(indices[k], indices[j]) = 1;
            }
        }
    }

    // constraints: g 和 h 的非零索引缓存
    auto process_constraints = [&](const std::vector<Eigen::MatrixXi>& supp) {
        for (const auto& B : supp) {
            Eigen::VectorXi check = Eigen::VectorXi::Zero(n);
            for (int row = 0; row < B.rows(); ++row) {
                for (int col = 0; col < B.cols(); ++col) {
                    if (B(row, col) > 0) {
                        check(B(row, col) - 1) = 1;
                    }
                }
            }
            for (int i = 0; i < n; ++i) {
                if (check(i)) {
                    for (int j = i + 1; j < n; ++j) {
                        if (check(j)) {
                            G(i, j) = 1;
                            G(j, i) = 1;
                        }
                    }
                }
            }
        }
    };

    // 处理 g 和 h
    process_constraints(supp_g);
    process_constraints(supp_h);

    return G;
}

// Chordal Extension
std::vector<Eigen::VectorXi> Polys::chordal_extension(const Eigen::MatrixXi& G, int mode) {
    Eigen::MatrixXi G_chordal;
    std::vector<int> order;
    if (mode == 1) {
        G_chordal = maximal_chordal_extension(G);
    } else if (mode == 2) {
        G_chordal = greedy_chordal_extension(G, order);
    } else if (mode == 3) {
        G_chordal = minimal_fill_chordal_extension(G, order);
    } else {
        throw std::invalid_argument("Invalid mode for chordal_extension.");
    }

    return find_max_cliques(G_chordal, order, mode == 1 ? 1 : 2);
}

// Record Constructions
Eigen::MatrixXi Polys::generate_all_sequences(int n, int d) {
    // Step 1: Initialize the first sequence (all zeros)
    Eigen::VectorXi seqs = Eigen::VectorXi::Zero(d);

    // Step 2: Calculate the total number of sequences
    int num_seqs = 1;
    for (int i = 1; i <= d; ++i) {
        num_seqs *= (n + i);
        num_seqs /= i;
    }

    // Step 3: Initialize the matrix to store all sequences
    Eigen::MatrixXi all_seqs(num_seqs, d);
    all_seqs.row(0) = seqs;

    // Step 4: Generate sequences iteratively
    for (int i = 1; i < num_seqs; ++i) {
        seqs = find_next_monomial(n, seqs, d);
        all_seqs.row(i) = seqs;
    }

    return all_seqs;
}

// Helper function: find_next_monomial
Eigen::VectorXi Polys::find_next_monomial(int n, const Eigen::VectorXi& J, int d) {
    Eigen::VectorXi next_seq = J;

    // Find the next valid sequence
    for (int i = d - 1; i >= 0; --i) {
        if (next_seq(i) < n) {
            int new_val = next_seq(i) + 1;
            for (int j = i; j < d; ++j) {
                next_seq(j) = new_val;
            }
            return next_seq;
        }
    }

    return next_seq; // Return the updated sequence
}


// build union vector
Eigen::MatrixXi Polys::build_union_vector(int n, int d, const Eigen::VectorXi& x_vars) {
    // Step 1: Adjust x_vars by adding 1
    Eigen::VectorXi adjusted_x_vars = x_vars.array() + 1;

    // Step 2: Add 1 to the beginning of x_vars
    Eigen::VectorXi extended_x_vars(adjusted_x_vars.size() + 1);
    extended_x_vars(0) = 1;
    extended_x_vars.tail(adjusted_x_vars.size()) = adjusted_x_vars;

    // Step 3: Generate all valid sequences
    Eigen::MatrixXi seqs = generate_all_sequences(n, d);
    seqs.array() += 1; // Shift to 1-based indexing
    int num_seqs = seqs.rows();
    Eigen::MatrixXi M(num_seqs, d);

    // Step 4: Vectorized mapping of seqs using x_vars
    for (int i = 0; i < num_seqs; ++i) {
        for (int j = 0; j < d; ++j) {
            M(i, j) = extended_x_vars(seqs(i, j) - 1); // Map using x_vars
        }
    }

    // Step 5: Restore original range by subtracting 1
    M.array() -= 1;

    return M;
}


bool Polys::check_even(const Eigen::VectorXi& vec) {
    int m = vec.rows();
    for (int i = 0; i < m; ) {
        if (vec[i] == 0) {  // 跳过 0 的处理
            ++i;
            continue;
        }

        int count = 1;
        while (i + 1 < m && vec[i] == vec[i + 1]) {
            ++count;
            ++i;
        }

        if (count % 2 != 0) {  // 非零元素出现次数为奇数
            return false;
        }

        ++i;  // 跳到下一个不同的元素
    }
    return true;
}


std::vector<uint64_t> Polys::get_key(const Eigen::MatrixXi& rpt, int n) {
    int m = rpt.rows(); 
    int d = rpt.cols(); 
    std::vector<uint64_t> keys(m, 0); 

    uint64_t base = n + 1; 

    for (int i = 0; i < m; ++i) { 
        uint64_t key_tmp = 0; 
        uint64_t multiplier = 1; 
        for (int j = d - 1; j >= 0; --j) { 
            key_tmp += rpt(i, j) * multiplier;
            multiplier *= base; 
        }
        keys[i] = key_tmp; 
    }

    return keys;
}

//merge sort
Eigen::RowVectorXi Polys::merge_sort(const Eigen::RowVectorXi& row1, const Eigen::RowVectorXi& row2) {
    
    Eigen::RowVectorXi result(row1.size() + row2.size());

    int i = 0, j = 0, k = 0;
    while (i < row1.size() && j < row2.size()) {
        if (row1[i] <= row2[j]) {
            result[k++] = row1[i++];
        } else {
            result[k++] = row2[j++];
        }
    }

    while (i < row1.size()) {
        result[k++] = row1[i++];
    }
    while (j < row2.size()) {
        result[k++] = row2[j++];
    }

    return result;
}


// support extension
Eigen::MatrixXi Polys::hs_support_extension_rpt(const std::vector<Eigen::MatrixXi>& mon_rpt, const std::unordered_map<uint64_t, int>& C, const Eigen::MatrixXi& supp_rpt_g, int idx_g, int mode) {
    Eigen::MatrixXi mon_tmp = mon_rpt[idx_g]; 
    int r = mon_tmp.rows(); 

    if (r == 1) {
        return Eigen::MatrixXi::Constant(1, 1, 1);
    }

    Eigen::MatrixXi B;

    if (mode == 0) {
        B = Eigen::MatrixXi::Zero(r, r); 

        for (int i = 1; i < r; ++i) { 
            for (int j = 0; j < i; ++j) {

                Eigen::MatrixXi combined(1, mon_tmp.cols() * 2);
                combined << mon_tmp.row(i), mon_tmp.row(j); 
                // sort
                std::sort(combined.data(), combined.data() + combined.size());
                std::vector<uint64_t> ij_key = get_key(combined, n);

                bool check = std::any_of(ij_key.begin(), ij_key.end(),
                                        [&](uint64_t key) { return C.find(key) != C.end(); });
                
                if (d != 0) {
                    if (!check) { 
                        for (int k = 0; k < combined.rows(); ++k) {
                            Eigen::VectorXi row = combined.row(k);
                            if (check_even(row)) {  
                                check = true;
                                break;  
                            }
                        }
                    }
                }

                B(i, j) = check ? 1 : 0;
            }
        }
        B = B.transpose() + B + Eigen::MatrixXi::Identity(r, r); 
        return B;
    }

    if (mode == 1) {
        B = Eigen::MatrixXi::Zero(r, r); 

        for (int i = 1; i < r; ++i) {
            for (int j = 0; j < i; ++j) {

                Eigen::MatrixXi combined(supp_rpt_g.rows(), supp_rpt_g.cols() + mon_tmp.cols() * 2);
                combined.block(0, 0, supp_rpt_g.rows(), mon_tmp.cols()) = mon_tmp.row(i).replicate(supp_rpt_g.rows(), 1);
                combined.block(0, mon_tmp.cols(), supp_rpt_g.rows(), mon_tmp.cols()) = mon_tmp.row(j).replicate(supp_rpt_g.rows(), 1);
                combined.block(0, 2 * mon_tmp.cols(), supp_rpt_g.rows(), supp_rpt_g.cols()) = supp_rpt_g;
                for (int k = 0; k < combined.rows(); ++k) {
                    Eigen::VectorXi row = combined.row(k);
                    std::sort(row.data(), row.data() + row.size());
                    combined.row(k) = row;
                }

                std::vector<uint64_t> ij_key = get_key(combined, n);

                bool check = std::any_of(ij_key.begin(), ij_key.end(),
                                         [&](uint64_t key) { return C.find(key) != C.end(); });
                if (d != 0) {
                    if (!check) { 
                        for (int k = 0; k < combined.rows(); ++k) {
                            Eigen::VectorXi row = combined.row(k);
                            if (check_even(row)) {  
                                check = true;
                                break;  
                            }
                        }
                    }
                }


                B(i, j) = check ? 1 : 0;
            }
        }
        B = B.transpose() + B + Eigen::MatrixXi::Identity(r, r); 
        return B;
    }

    if (mode == 2) {
        B = Eigen::MatrixXi::Zero(r, 1); 

        for (int i = 0; i < r; ++i) {

            Eigen::MatrixXi combined(supp_rpt_g.rows(), supp_rpt_g.cols() + mon_tmp.cols());
            combined.block(0, 0, supp_rpt_g.rows(), mon_tmp.cols()) = mon_tmp.row(i).replicate(supp_rpt_g.rows(), 1);
            combined.block(0, mon_tmp.cols(), supp_rpt_g.rows(), supp_rpt_g.cols()) = supp_rpt_g;
            for (int k = 0; k < combined.rows(); ++k) {
                Eigen::VectorXi row = combined.row(k);
                std::sort(row.data(), row.data() + row.size());
                combined.row(k) = row;
            }

            std::vector<uint64_t> i_key = get_key(combined, n);

            bool check = std::any_of(i_key.begin(), i_key.end(),
                                     [&](uint64_t key) { return C.find(key) != C.end(); });
            if (d != 0) {
                if (!check) { 
                    for (int k = 0; k < combined.rows(); ++k) {
                        Eigen::VectorXi row = combined.row(k);
                        if (check_even(row)) {  
                            check = true;
                            break;  
                        }
                    }
                }
            }

            B(i, 0) = check ? 1 : 0;
        }

        return B;
    }

    return B;
}

//block closure
int Polys::find_root(int i, std::vector<int>& parent) {
    if (parent[i] != i) {
        parent[i] = find_root(parent[i], parent); 
    }
    return parent[i];
}

Eigen::MatrixXi Polys::block_closure(const Eigen::MatrixXi& B) {
    int r = B.rows(); 
    Eigen::MatrixXi B_new = B; 

    std::vector<int> parent(r);
    for (int i = 0; i < r; ++i) {
        parent[i] = i; 
    }

    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < r; ++j) {
            if (B(i, j) == 1) {
                int root_i = find_root(i, parent);
                int root_j = find_root(j, parent);
                if (root_i != root_j) {
                    parent[root_j] = root_i; 
                }
            }
        }
    }

    for (int i = 0; i < r; ++i) {
        int root_i = find_root(i, parent); 
        for (int j = 0; j < r; ++j) {
            int root_j = find_root(j, parent); 
            if (root_i == root_j) { 
                B_new(i, j) = 1; 
            }
        }
    }

    return B_new;
}

// find blocks in term sparsity
std::vector<Eigen::VectorXi> Polys::find_blocks(const Eigen::MatrixXi& B) {
    int r = B.rows(); 
    std::vector<bool> visited(r, false); 
    std::vector<Eigen::VectorXi> I_ele; 

    for (int i = 0; i < r; ++i) {
        if (!visited[i]) { 
            std::vector<int> blockCols; 

            for (int j = 0; j < B.cols(); ++j) {
                if (B(i, j) != 0) {
                    blockCols.push_back(j); 
                }
            }

            Eigen::VectorXi blockColsVec = Eigen::VectorXi::Map(blockCols.data(), blockCols.size());
            I_ele.push_back(blockColsVec); 

            for (int col : blockCols) {
                for (int row = 0; row < r; ++row) {
                    if (B(row, col) != 0) {
                        visited[row] = true;
                    }
                }
            }
        }
    }

    return I_ele;
}

// Moment and SOS
std::vector<int> Polys::find_loca_1(int order, std::vector<int>& sizes) {
    std::vector<int> index;
    // Total index to [cell index, local index]
    int cellIdx = 0; // Index starts from 0 in C++
    while (order >= sizes[cellIdx]) {
        order -= sizes[cellIdx];
        cellIdx++;
    }
    index.push_back(cellIdx); 
    index.push_back(order);
    return index;
}

int Polys::find_loca_2(std::vector<int> order, std::vector<int>& sizes) {
    int index = 0;
    for (int i = 0; i < order[0]; ++i) {
        index += sizes[i];
    }
    index += order[1];
    return index;
}

Eigen::MatrixXi Polys::generate_2N(int n, int d) {
    Eigen::MatrixXi result;

    if (d == 2) {
        int total = 0;
        for (int i = 0; i < cI.size(); ++i) {
            int tmp = cI[i].rows();
            total += (tmp + 1) * (tmp + 2) / 2;
        }
        result.resize(total, 4);
        int index = 0;
        for (int k = 0; k < cI.size(); ++k) {
            for (int i = 0; i < cI[k].rows(); ++i) {
                for (int j = i; j < cI[k].rows(); ++j) {
                    result(index, 0) = cI[k](i);
                    result(index, 1) = cI[k](i);
                    result(index, 2) = cI[k](j);
                    result(index, 3) = cI[k](j);
                    index++;
                }
            }
        }
    }

    if (d == 3) {
        int total = 0;
        for (int i = 0; i < cI.size(); ++i) {
            int tmp = cI[i].rows();
            total += (tmp + 1) * (tmp + 2) * (tmp + 3) / 6;
        }
        result.resize(total, 6);
        int index = 0;
        for (int k = 0; k < cI.size(); ++k) {
            for (int i = 0; i < cI[k].rows(); ++i) {
                for (int j = i; j < cI[k].rows(); ++j) {
                    for (int l = j; l < cI[k].rows(); ++l) {
                        result(index, 0) = cI[k](i);
                        result(index, 1) = cI[k](i);
                        result(index, 2) = cI[k](j);
                        result(index, 3) = cI[k](j);
                        result(index, 4) = cI[k](l);
                        result(index, 5) = cI[k](l);
                        index++;
                    }
                }
            }
        }
    }

    return result;
}


























// debug function
void Polys::printdebug(Eigen::MatrixXi& G) const {
    std::cout<< "Matrix" << std::endl;
    for (int i = 0; i < G.rows(); ++i) {
        for (int j = 0; j < G.cols(); ++j) {
            std::cout << G(i, j) << " ";
        }
        std::cout << std::endl;
    } 
    std::cout << std::endl;   
}

void Polys::printdebug2(Eigen::VectorXi& G) const {
    std::cout<< "Vector" << std::endl;
    for (int i = 0; i<G.rows();++i){
        std::cout<<G(i)<<" ";
    }
    std::cout<<std::endl;
}

void Polys::printtI() const {
    for (size_t i = 0; i < tI.size(); ++i) {
        std::cout << "Block " << i + 1 << ":" << std::endl;
        const auto& block = tI[i];
        for (size_t j = 0; j < block.size(); ++j) {
            std::cout << "  Sub-block " << j + 1 << ": [";
            for (int k = 0; k < block[j].size(); ++k) {
                std::cout << block[j][k];
                if (k != block[j].size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;
        }
        std::cout << std::endl;
    }
}

void Polys::printkey(std::unordered_map<uint64_t, int> alpha) const {
    std::vector<uint64_t> keys;
    for (const auto& pair : alpha) {
        keys.push_back(pair.first);
    }

    std::sort(keys.begin(), keys.end());

    std::cout << "Keys in C (sorted):" << std::endl;
    for (const auto& key : keys) {
        std::cout << key << std::endl;
    }
}

void Polys::printmoment() const {
    std::cout<< A_moment.rows()<<" "<<A_moment.cols()<<std::endl;
    std::cout<< C_moment.rows()<<" "<<C_moment.cols()<<std::endl;
    std::cout<< b_moment.rows()<<" "<<b_moment.cols()<<std::endl;
}

void Polys::printsos() const {
    std::cout<< A_sos.rows()<<" "<<A_sos.cols()<<std::endl;
    std::cout<< a_sos.rows()<<" "<<a_sos.cols()<<std::endl;
    std::cout<< b_sos.rows()<<" "<<b_sos.cols()<<std::endl;
}