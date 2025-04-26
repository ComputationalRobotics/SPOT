#include "gen_test.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>

void initializePolys(Polys& problem) {
    // --- Step 1: Define the objective polynomial f ---
    // f = x1^2 + x2^2 + x3^2 + x4^2 + x5^2 + x6^2 + x7^2
    Eigen::MatrixXi supp_f(7, 4);  // 7 terms, max support size is 4
    supp_f << 0, 0, 1, 1,  // x1^2
              0, 0, 2, 2,  // x2^2
              0, 0, 3, 3,  // x3^2
              0, 0, 4, 4,  // x4^2
              0, 0, 5, 5,  // x5^2
              0, 0, 6, 6,  // x6^2
              0, 0, 7, 7;  // x7^2
    Eigen::VectorXd coeff_f(7);
    coeff_f << 1, 1, 1, 1, 1, 1, 1;  // All coefficients are 1

    problem.setObjective(supp_f, coeff_f, 7, 2);

    // --- Step 2: Define the inequality constraints g ---
    // g = [1 - x2^2; 1 - x4^2; 1 - x6^2]
    int m_ineq = 3;
    std::vector<Eigen::MatrixXi> supp_g;
    std::vector<Eigen::VectorXd> coeff_g;

    // Constraint 1: 1 - x2^2
    Eigen::MatrixXi supp_g1(2, 4);  // 2 terms, max support size is 4
    supp_g1 << 0, 0, 0, 0,  // constant terms
               0, 0, 2, 2;  // x2^2
    Eigen::VectorXd coeff_g1(2);
    coeff_g1 << 1, -1;

    supp_g.push_back(supp_g1);
    coeff_g.push_back(coeff_g1);

    // Constraint 2: 1 - x4^2
    Eigen::MatrixXi supp_g2(2, 4);
    supp_g2 << 0, 0, 0, 0,  // constant term
               0, 0, 4, 4;  // x4^2
    Eigen::VectorXd coeff_g2(2);
    coeff_g2 << 1, -1;

    supp_g.push_back(supp_g2);
    coeff_g.push_back(coeff_g2);

    // Constraint 3: 1 - x6^2
    Eigen::MatrixXi supp_g3(2, 4);
    supp_g3 << 0, 0, 0, 0,  // constant term
               0, 0, 6, 6;  // x6^2
    Eigen::VectorXd coeff_g3(2);
    coeff_g3 << 1, -1;

    supp_g.push_back(supp_g3);
    coeff_g.push_back(coeff_g3);

    std::vector<int> dj_g = {1, 1, 1};  // Degrees of the inequality constraints
    problem.setInequalities(supp_g, coeff_g, dj_g);

    // --- Step 3: Define the equality constraints h ---
    // h = [x3 - 0.9*x1 + 0.1*x1*x2;
    //      x5 - 0.9*x3 + 0.1*x3*x4;
    //      x7 - 0.9*x5 + 0.1*x5*x6;
    //      x1 - 2]
    int m_eq = 4;
    std::vector<Eigen::MatrixXi> supp_h;
    std::vector<Eigen::VectorXd> coeff_h;

    // Constraint 1: x3 - 0.9*x1 + 0.1*x1*x2
    Eigen::MatrixXi supp_h1(3, 4);
    supp_h1 << 0, 0, 0, 3,  // x3
               0, 0, 0, 1,  // -0.9*x1
               0, 0, 1, 2;  // 0.1*x1*x2
    Eigen::VectorXd coeff_h1(3);
    coeff_h1 << 1, -0.9, 0.1;

    supp_h.push_back(supp_h1);
    coeff_h.push_back(coeff_h1);

    // Constraint 2: x5 - 0.9*x3 + 0.1*x3*x4
    Eigen::MatrixXi supp_h2(3, 4);
    supp_h2 << 0, 0, 0, 5,  // x5
               0, 0, 0, 3,  // -0.9*x3
               0, 0, 3, 4;  // 0.1*x3*x4
    Eigen::VectorXd coeff_h2(3);
    coeff_h2 << 1, -0.9, 0.1;

    supp_h.push_back(supp_h2);
    coeff_h.push_back(coeff_h2);

    // Constraint 3: x7 - 0.9*x5 + 0.1*x5*x6
    Eigen::MatrixXi supp_h3(3, 4);
    supp_h3 << 0, 0, 0, 7,  // x7
               0, 0, 0, 5,  // -0.9*x5
               0, 0, 5, 6;  // 0.1*x5*x6
    Eigen::VectorXd coeff_h3(3);
    coeff_h3 << 1, -0.9, 0.1;

    supp_h.push_back(supp_h3);
    coeff_h.push_back(coeff_h3);

    // Constraint 4: x1 - 2
    Eigen::MatrixXi supp_h4(2, 4);
    supp_h4 << 0, 0, 0, 1,  // x1
               0, 0, 0, 0;  // constant term
    Eigen::VectorXd coeff_h4(2);
    coeff_h4 << 1, -2;

    supp_h.push_back(supp_h4);
    coeff_h.push_back(coeff_h4);

    std::vector<int> dj_h = {2, 2, 2, 1};  // Degrees of the equality constraints
    problem.setEqualities(supp_h, coeff_h, dj_h);

    // Done initializing the problem
}