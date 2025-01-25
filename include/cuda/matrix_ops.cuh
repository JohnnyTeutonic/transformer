#pragma once

#include "../matrix.hpp"

#ifdef USE_CUDA
namespace cuda {
    // Initialize and cleanup CUDA resources
    void initialize_cuda();
    void cleanup_cuda();

    // Matrix operations
    Matrix matmul(const Matrix& A, const Matrix& B);
    void gelu_forward(Matrix& x);

    // Beam search operations
    void topk(const Matrix& input, Matrix& top_k_values, ::std::vector<int>& top_k_indices, int k);
    void topk(const ::std::vector<float>& input, Matrix& top_k_values, ::std::vector<int>& top_k_indices, int k);
    void beam_search_step(const Matrix& model_output, const Matrix& beam_scores,
                         Matrix& next_scores, ::std::vector<int>& next_tokens,
                         int beam_width);
}
#endif // USE_CUDA 