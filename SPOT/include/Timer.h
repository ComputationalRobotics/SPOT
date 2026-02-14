#ifndef TIMER_H
#define TIMER_H

#include <iostream>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

class Timer {
public:
    // Start timer
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    // Stop timer and record elapsed time
    void stop(const std::string& label) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time; // Default unit: seconds
        std::cout << label << " execution time: " << elapsed.count() << " s" << std::endl;
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
};

#endif