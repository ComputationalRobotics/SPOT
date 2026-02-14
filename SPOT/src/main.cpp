#include "gen_test.h"  // Header containing the initializePolys function
#include "Polys.h"     // Polys class declaration and implementation
#include "Timer.h"
#include <iostream>

int main() {
    // Create Polys object
    Polys problem;
    Timer timer;

    // Initialize the problem using initializePolys
    initializePolys(problem);
    problem.printInfo();


    // Compute correlative sparsity cliques (cI)
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