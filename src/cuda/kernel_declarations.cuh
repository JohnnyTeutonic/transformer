namespace cuda {
    __global__ void attention_scores_kernel(const float* queries, const float* keys,
                                          float* scores, float scale,
                                          int seq_len, int head_dim);
    
    __global__ void softmax_kernel(float* matrix, int rows, int cols);
    
    __global__ void attention_kernel(const float* Q, const float* K, const float* V,
                                   float* output, int batch_size, int seq_len, 
                                   int head_dim, int hidden_dim);
                                   
    __global__ void scaled_dot_product_attention_kernel(const float* Q, const float* K,
                                                       const float* V, float* output,
                                                       const float* mask, int batch_size,
                                                       int num_heads, int seq_len,
                                                       int head_dim, float scale);
} 