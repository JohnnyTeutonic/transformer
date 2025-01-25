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
    float dropout_rate;           ///< Probability of dropping a unit
    mutable Matrix mask_cache;      ///< Cached dropout mask

  public:
    /**
     * @brief Constructs a dropout layer.
     * @param rate Probability of dropping each unit (between 0 and 1)
     */
    explicit Dropout(float rate = 0.1f) : dropout_rate(rate) {}

    /**
     * @brief Performs the forward pass with dropout.
     * 
     * During training, randomly drops units with probability dropout_rate and
     * scales remaining units by 1/(1-dropout_rate). During inference, performs
     * no dropout and no scaling.
     * 
     * @param input Input matrix to apply dropout to
     * @param training Whether in training mode (true) or inference mode (false)
     * @return Matrix with dropout applied
     * @throws std::runtime_error if dimensions mismatch between input and mask
     */
    Matrix forward(const Matrix& input, bool training = true) const {
        if (!training || dropout_rate == 0.0f) {
            return input;
        }

        Matrix dropout_mask(input.rows(), input.cols());
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution d(1.0f - dropout_rate);

        for (size_t i = 0; i < dropout_mask.size(); ++i) {
            dropout_mask.get_data()[i] = d(gen) / (1.0f - dropout_rate);
        }

        mask_cache = dropout_mask;
        return input.hadamard(dropout_mask);
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
        if (grad_output.rows() != mask_cache.rows() ||
            grad_output.cols() != mask_cache.cols()) {
            throw std::runtime_error("Gradient dimensions (" + std::to_string(grad_output.rows()) +
                                     "," + std::to_string(grad_output.cols()) +
                                     ") don't match dropout mask dimensions (" +
                                     std::to_string(mask_cache.rows()) + "," +
                                     std::to_string(mask_cache.cols()) + ")");
        }

        return grad_output.hadamard(mask_cache);
    }

    /**
     * @brief Gets the dimensions of the current dropout mask.
     * @return Pair of (rows, columns) representing mask dimensions
     */
    std::pair<size_t, size_t> get_mask_dimensions() const {
        return {mask_cache.rows(), mask_cache.cols()};
    }
};