#pragma once
#include <vector>

struct EvaluationMetrics {
    float accuracy;
    float precision;
    float recall;
    float f_score;
};

EvaluationMetrics compute_metrics(const std::vector<int>& predictions, const std::vector<int>& targets); 