#define _GLIBCXX_USE_CXX11_ABI 0   // Align the ABI version
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "ObjectiveFunction.h"
#include "Point.h"
#include <vector>
#include <cstddef>

/**
 * @brief Base class for optimizers
 */
class BaseOptimizer {
public:
    ///////////////////////////////////
    // Constructors
    ///////////////////////////////////
    BaseOptimizer(BaseFunction &obj,
                  std::vector<Point2<double>> &var)
        : obj_(obj), var_(var) {}

    ///////////////////////////////////
    // Methods
    ///////////////////////////////////

    /// Prepare internal buffers, etc.
    virtual void Initialize() = 0;

    /**
     * @brief Perform one optimization step.
     * @return status code (0 = success, >0 = special condition / error).
     */
    virtual int Step() = 0;

protected:
    ///////////////////////////////////
    // Data members
    ///////////////////////////////////
    BaseFunction &obj_;                 ///< Objective function to optimize
    std::vector<Point2<double>> &var_;  ///< Variables to optimize
};

/**
 * @brief Simple Conjugate-Gradient optimizer
 */
class SimpleConjugateGradient : public BaseOptimizer {
public:
    ///////////////////////////////////
    // Constructors
    ///////////////////////////////////
    SimpleConjugateGradient(BaseFunction               &obj,
                            std::vector<Point2<double>> &var,
                            const double               &alpha);

    ///////////////////////////////////
    // Methods
    ///////////////////////////////////
    void Initialize() override;
    int  Step()       override;   ///< Now returns status code

private:
    ///////////////////////////////////
    // Data members
    ///////////////////////////////////
    std::vector<Point2<double>> grad_prev_;  ///< g_{k-1}
    std::vector<Point2<double>> dir_prev_;   ///< d_{k-1}
    std::size_t                 step_;       ///< Iteration counter
    double                      alpha_;      ///< Initial line-search step
};

#endif  // OPTIMIZER_H
