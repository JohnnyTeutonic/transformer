#pragma once
#include "matrix.hpp"
#include <random>

/**
 * @brief Implements dropout regularization for neural networks.
 * 
 * The Dropout class provides a mechanism for randomly "dropping out" units during training,
 * which helps prevent overfitting. Features include:
 * - Configurable dropout rate
 * - Training/inference mode switching
 * - Automatic scaling during training
 * - Mask caching for backpropagation
 */
class Dropout {
  private:
    float dropout_prob;
    Matrix dropout_mask;
    std::mt19937 gen{std::random_device{}()};
    bool training = true;

  public:
    /**
     * @brief Constructs a dropout layer.
     * @param prob Probability of dropping each unit (between 0 and 1)
     */
    explicit Dropout(float prob) : dropout_prob(prob) {}

    /**
     * @brief Performs the forward pass with dropout.
     * 
     * During training, randomly drops units with probability dropout_prob and
     * scales remaining units by 1/(1-dropout_prob). During inference, performs
     * no dropout and no scaling.
     * 
     * @param input Input matrix to apply dropout to
     * @return Matrix with dropout applied
     * @throws std::runtime_error if dimensions mismatch between input and mask
     */
    Matrix forward(Matrix& input) {
        if (!training || dropout_prob == 0.0f) {
            return input;
        }

        // Create dropout mask with same dimensions as input
        dropout_mask = Matrix(input.rows(), input.cols(), 1.0f);
        std::bernoulli_distribution dist(1.0f - dropout_prob);

        // Apply dropout mask
        for (size_t i = 0; i < dropout_mask.size(); i++) {
            if (!dist(gen)) {
                dropout_mask.data()[i] = 0.0f;
            }
        }

        // Scale output by dropout probability
        Matrix output = input;
        for (size_t i = 0; i < output.size(); i++) {
            output.data()[i] *= dropout_mask.data()[i] / (1.0f - dropout_prob);
        }

        return output;
    }

    /**
     * @brief Performs the backward pass of dropout.
     * 
     * Applies the same dropout mask from the forward pass to the gradient,
     * ensuring consistent gradient flow through the network.
     * 
     * @param grad_output Gradient of the loss with respect to the output
     * @return Gradient with respect to the input
     * @throws std::runtime_error if mask not initialized or dimensions mismatch
     */
    Matrix backward(const Matrix& grad_output) const {
        if (!training) {
            throw std::runtime_error(
                "Dropout mask not initialized. Forward pass must be called before backward pass");
        }

        if (grad_output.rows() != dropout_mask.rows() ||
            grad_output.cols() != dropout_mask.cols()) {
            throw std::runtime_error("Gradient dimensions (" + std::to_string(grad_output.rows()) +
                                     "," + std::to_string(grad_output.cols()) +
                                     ") don't match dropout mask dimensions (" +
                                     std::to_string(dropout_mask.rows()) + "," +
                                     std::to_string(dropout_mask.cols()) + ")");
        }

        return grad_output.hadamard(dropout_mask);
    }

    /**
     * @brief Gets the dimensions of the current dropout mask.
     * @return Pair of (rows, columns) representing mask dimensions
     */
    std::pair<size_t, size_t> get_mask_dimensions() const {
        return {dropout_mask.rows(), dropout_mask.cols()};
    }

    void set_training(bool mode) {
        training = mode;
    }

    void reset_mask() {
        dropout_mask = Matrix();
    }
};