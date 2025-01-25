#include "../include/components.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>

void print_matrix_stats(const Matrix& m) {
    std::cout << "Matrix Stats:\n"
              << "Dimensions: " << m.rows() << "x" << m.cols() << "\n"
              << "Size: " << m.size() << " elements\n"
              << "Memory: " << (m.size() * sizeof(float)) / 1024.0f << " KB\n";
    
    if (!m.empty()) {
        float min_val = m.min();
        float max_val = m.max();
        std::cout << "Range: [" << min_val << ", " << max_val << "]\n";
    }
}
