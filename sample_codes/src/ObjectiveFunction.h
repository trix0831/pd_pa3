#define _GLIBCXX_USE_CXX11_ABI 0
#ifndef OBJECTIVEFUNCTION_H
#define OBJECTIVEFUNCTION_H

#include <vector>
#include "Placement.h"
#include "Point.h"

/**
 * @brief Base class for objective functions
 */
class BaseFunction {
public:
    explicit BaseFunction(std::size_t input_size)
        : grad_(input_size), value_(0.0) {}

    const std::vector<Point2<double>>& grad() const { return grad_; }
    const double& value() const { return value_; }

    // Virtual interface
    virtual const double& operator()(const std::vector<Point2<double>>& input) = 0;
    virtual const std::vector<Point2<double>>& Backward() = 0;

    // NEW: virtual lambda setter (no-op default)
    virtual void setLambda(double) {}

protected:
    std::vector<Point2<double>> grad_;
    double value_;
};

/**
 * @brief Example function for optimization
 */
class ExampleFunction : public BaseFunction {
public:
    ExampleFunction(Placement& placement, double lambda = 1.0);

    const double& operator()(const std::vector<Point2<double>>& input) override;
    const std::vector<Point2<double>>& Backward() override;

    void setLambda(double lambda) override { lambda_ = lambda; }

private:
    std::vector<Point2<double>> input_;
    Placement& placement_;
    double lambda_;
};

/**
 * @brief Wirelength function (stub)
 */
class Wirelength : public BaseFunction {
public:
    explicit Wirelength(Placement& pl)
        : BaseFunction(pl.numModules()), placement_(pl) {}

    const double& operator()(const std::vector<Point2<double>>& input) override;
    const std::vector<Point2<double>>& Backward() override;

private:
    Placement& placement_;
};

/**
 * @brief Density function (stub)
 */
class Density : public BaseFunction {
public:
    explicit Density(Placement& pl)
        : BaseFunction(pl.numModules()), placement_(pl) {}

    const double& operator()(const std::vector<Point2<double>>& input) override;
    const std::vector<Point2<double>>& Backward() override;

private:
    Placement& placement_;
};

/**
 * @brief Objective function: wirelength + lambda × density
 */
class ObjectiveFunction : public BaseFunction {
public:
    ObjectiveFunction(Placement& pl, double lambda = 1.0)
        : BaseFunction(pl.numModules()),
          wirelength_(pl),
          density_(pl),
          lambda_(lambda) {}

    const double& operator()(const std::vector<Point2<double>>& input) override;
    const std::vector<Point2<double>>& Backward() override;

    void setLambda(double lambda) override { lambda_ = lambda; }

private:
    Wirelength wirelength_;
    Density density_;
    double lambda_;
};

#endif  // OBJECTIVEFUNCTION_H
