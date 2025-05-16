#include "Optimizer.h"

#include <cmath>

SimpleConjugateGradient::SimpleConjugateGradient(BaseFunction &obj,
                                                 std::vector<Point2<double>> &var,
                                                 const double &alpha,
                                                 Placement &placement)
    : BaseOptimizer(obj, var),
      grad_prev_(var.size()),
      dir_prev_(var.size()),
      step_(0),
      alpha_(alpha),
      placement_(placement),
      x_step_max((placement.boundryRight()-placement.boundryLeft())/50),
      y_step_max((placement.boundryTop()-placement.boundryBottom())/50){}

void SimpleConjugateGradient::Initialize() {
    // Before the optimization starts, we need to initialize the optimizer.
    step_ = 0;
}

/**
 * @details Update the solution once using the conjugate gradient method.
 */
double SimpleConjugateGradient::Step() {
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
    // Make dir[i] a unit vector
    for (size_t i = 0; i < kNumModule; ++i) {
        double norm = std::sqrt(dir[i].x * dir[i].x + dir[i].y * dir[i].y);
        if (norm > 1e-12) {
            dir[i].x /= norm;
            dir[i].y /= norm;
        }
    }

    double count = 0;

    double outerScale = 0.8;
    double height = placement_.boundryTop()-placement_.boundryBottom();
    double width = placement_.boundryRight()-placement_.boundryLeft();
    // printf("height: %f, width: %f",height, width);`

    double top = placement_.boundryTop() - height*(1-outerScale)/2;
    double bottom = placement_.boundryBottom() + height*(1-outerScale)/2;
    double right = placement_.boundryRight() - width*(1-outerScale)/2;
    double left = placement_.boundryLeft() + width*(1-outerScale)/2;
    
    // printf("top: %f, bottom: %f, right: %f, left: %f\n", top, bottom, right, left);

    // Update the solution
    for (size_t i = 0; i < kNumModule; ++i) {
        var_[i].x = var_[i].x + x_step_max * dir[i].x;  // Update the variable (solution)
        var_[i].y = var_[i].y + y_step_max * dir[i].y;

        // count how many module out of the outer, which shows the spreading process
        if (var_[i].x < left || var_[i].x > right || var_[i].y < bottom || var_[i].y > top) {
            // printf("module[%lu]: (%f,%f) out of bound\n", i, var_[i].x, var_[i].y);
            count++;
        }
    }

    printf("out of bound percent: %f\n", (double)count / (double)kNumModule);

    // Update the cache for the next iteration
    grad_prev_ = obj_.grad();
    dir_prev_ = dir;
    step_++;

    count = (double)count / (double)kNumModule;
    // the larger lambda is, the more weight on the density
    obj_.setLambda(1-count);
    printf("lambda: %f\n", 1-count);

    return count;  // Return the percentage of modules out of bounds
}
