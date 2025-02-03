#include "../include/evaluation_metrics.hpp"

EvaluationMetrics compute_metrics(const std::vector<int>& predictions, const std::vector<int>& targets) {
    size_t correct = 0;
    size_t true_positives = 0;
    size_t false_positives = 0;
    size_t false_negatives = 0;

    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == targets[i]) {
            correct++;
            true_positives++;
        } else {
            if (predictions[i] != -1) false_positives++;
            if (targets[i] != -1) false_negatives++;
        }
    }

    float accuracy = static_cast<float>(correct) / predictions.size();
    float precision = true_positives > 0 ? 
        static_cast<float>(true_positives) / (true_positives + false_positives) : 0.0f;
    float recall = true_positives > 0 ? 
        static_cast<float>(true_positives) / (true_positives + false_negatives) : 0.0f;
    float f_score = (precision + recall) > 0 ? 
        2 * (precision * recall) / (precision + recall) : 0.0f;

    return {accuracy, precision, recall, f_score};
} 