#include "gen_test.h"  // 包含 initializePolys 函数的头文件
#include "Polys.h"     // 假设 Polys 类的声明和实现位于该头文件及其对应的 .cpp 文件中
#include "Timer.h"
#include <iostream>

int main() {
    // 创建 Polys 对象
    Polys problem;
    Timer timer;

    // 使用 initializePolys 函数初始化问题
    initializePolys(problem);
    problem.printInfo();


    // 计算问题的 correlative sparsity cliques (cI)
    problem.Gen_cI("MD");
    
    //assign constraints
    timer.start();
    problem.Assign_Constraints();
    timer.stop("exampleFunction1");
    
    //Record Constraints
    problem.Record_monomials();

    //Construct C
    problem.Construct_C();         
    
    // Term Sparsity
    problem.Gen_tI("MAX", "NON");
    problem.printtI();

    // Moment Conversion
    problem.Moment_Conversion("USE");
    problem.printmoment();

    problem.SOS_Conversion("USE");
    problem.printsos();

    problem.CSTSS_Real({},{},{},{}, true);

    return 0;
}