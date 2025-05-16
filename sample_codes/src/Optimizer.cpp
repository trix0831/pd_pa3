#include "Optimizer.h"

#include <cmath>

SimpleConjugateGradient::SimpleConjugateGradient(BaseFunction &obj,
                                                 std::vector<Point2<double>> &var,
                                                 const double &alpha)
    : BaseOptimizer(obj, var),
      grad_prev_(var.size()),
      dir_prev_(var.size()),
      step_(0),
      alpha_(alpha) {}

void SimpleConjugateGradient::Initialize() {
    // Before the optimization starts, we need to initialize the optimizer.
    step_ = 0;
}

/**
 * @details Update the solution once using the conjugate gradient method.
 */
int SimpleConjugateGradient::Step() {
    const size_t &kNumModule = var_.size();

    // Compute the gradient direction
    obj_(var_);       // Forward pass: Compute the function value and cache from the input
    obj_.Backward();  // Backward pass: Compute the gradient according to the cache

    // Polak-Ribiere Conjugate Gradient Direction
    double beta = 0.0;                         // Polak-Ribiere coefficient
    std::vector<Point2<double>> dir(kNumModule);  // Conjugate directions

    if (step_ == 0) {
        // For the first step, we set beta = 0 and d_0 = -g_0
        for (size_t i = 0; i < kNumModule; ++i) {
            dir[i] = -obj_.grad().at(i);  // Initial direction is the negative gradient
        }
    } else {
        // For subsequent steps, calculate the Polak-Ribiere coefficient and conjugate directions
        double t1 = 0.0;  // Numerator for beta calculation
        double t2 = 0.0;  // Denominator for beta calculation
        const double epsilon = 1e-12;

        for (size_t i = 0; i < kNumModule; ++i) {
            Point2<double> grad_diff = obj_.grad().at(i) - grad_prev_.at(i);
            t1 += grad_diff.x * obj_.grad().at(i).x + grad_diff.y * obj_.grad().at(i).y;
            t2 += std::abs(obj_.grad().at(i).x) + std::abs(obj_.grad().at(i).y);
        }

        if (t2 < epsilon) {
            beta = 0.0;  // Small gradient: Avoid division by zero
        } else {
            beta = t1 / (t2 * t2);  // Polak-Ribiere coefficient
        }

        // Update the conjugate direction with beta
        for (size_t i = 0; i < kNumModule; ++i) {
            dir[i] = -obj_.grad().at(i) + beta * dir_prev_.at(i);  // Update direction
        }
    }

    // Step size (alpha) can be optimized dynamically; for now, use a fixed value
    // We can explore line search or dynamic step size in the future
    double step_size = alpha_;  // Fixed step size for now

    // Update the solution
    for (size_t i = 0; i < kNumModule; ++i) {
        var_[i] = var_[i] + step_size * dir[i];  // Update the variable (solution)
    }

    // Update the cache for the next iteration
    grad_prev_ = obj_.grad();
    dir_prev_ = dir;
    step_++;

    return 0;
}
