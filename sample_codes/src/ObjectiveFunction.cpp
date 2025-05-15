#include "ObjectiveFunction.h"

#include "cstdio"

ExampleFunction::ExampleFunction(Placement &placement) : BaseFunction(1), placement_(placement)
{
    printf("Fetch the information you need from placement database.\n");
    printf("For example:\n");
    printf("    Placement boundary: (%.f,%.f)-(%.f,%.f)\n", placement_.boundryLeft(), placement_.boundryBottom(),
           placement_.boundryRight(), placement_.boundryTop());
}

const double &ExampleFunction::operator()(const std::vector<Point2<double>> &input)
{
    value_ = 0.0;
    input_ = input;
    const double alpha = 1.0;

    for (unsigned i = 0; i < placement_.numNets(); ++i) {
        Net& net = placement_.net(i);
        std::vector<double> xs, ys;

        // Collect coordinates
        for (unsigned j = 0; j < net.numPins(); ++j) {
            Pin& pin = net.pin(j);
            const Point2<double>& pt = input[pin.moduleId()];
            xs.push_back(pt.x);
            ys.push_back(pt.y);
        }

        // Compute LSE x
        double xmax = *std::max_element(xs.begin(), xs.end());
        double xmin = *std::min_element(xs.begin(), xs.end());
        double sumX = 0, sumNegX = 0;
        for (double x : xs) {
            sumX    += std::exp((x - xmax) / alpha);
            sumNegX += std::exp((xmin - x) / alpha);
        }
        double lseX = alpha * (std::log(sumX) + xmax / alpha + std::log(sumNegX) + xmin / alpha);

        // Compute LSE y
        double ymax = *std::max_element(ys.begin(), ys.end());
        double ymin = *std::min_element(ys.begin(), ys.end());
        double sumY = 0, sumNegY = 0;
        for (double y : ys) {
            sumY    += std::exp((y - ymax) / alpha);
            sumNegY += std::exp((ymin - y) / alpha);
        }
        double lseY = alpha * (std::log(sumY) + ymax / alpha + std::log(sumNegY) + ymin / alpha);

        value_ += lseX + lseY;
    }

    return value_;
}




const std::vector<Point2<double>> &ExampleFunction::Backward()
{
    grad_.assign(input_.size(), Point2<double>(0.0, 0.0));
    const double alpha = 0.01;

    for (unsigned i = 0; i < placement_.numNets(); ++i) {
        Net& net = placement_.net(i);
        unsigned npins = net.numPins();
        std::vector<unsigned> moduleIds(npins);
        std::vector<double> xs(npins), ys(npins);

        for (unsigned j = 0; j < npins; ++j) {
            Pin& pin = net.pin(j);
            moduleIds[j] = pin.moduleId();
            xs[j] = input_[moduleIds[j]].x;
            ys[j] = input_[moduleIds[j]].y;
        }

        // X gradient
        double xmax = *std::max_element(xs.begin(), xs.end());
        double xmin = *std::min_element(xs.begin(), xs.end());
        std::vector<double> exp_x(npins), exp_neg_x(npins);
        double sumExpX = 0.0, sumExpNegX = 0.0;
        for (unsigned j = 0; j < npins; ++j) {
            exp_x[j] = std::exp((xs[j] - xmax) / alpha);
            exp_neg_x[j] = std::exp((xmin - xs[j]) / alpha);
            sumExpX += exp_x[j];
            sumExpNegX += exp_neg_x[j];
        }

        // Y gradient
        double ymax = *std::max_element(ys.begin(), ys.end());
        double ymin = *std::min_element(ys.begin(), ys.end());
        std::vector<double> exp_y(npins), exp_neg_y(npins);
        double sumExpY = 0.0, sumExpNegY = 0.0;
        for (unsigned j = 0; j < npins; ++j) {
            exp_y[j] = std::exp((ys[j] - ymax) / alpha);
            exp_neg_y[j] = std::exp((ymin - ys[j]) / alpha);
            sumExpY += exp_y[j];
            sumExpNegY += exp_neg_y[j];
        }

        // Accumulate gradients
        for (unsigned j = 0; j < npins; ++j) {
            unsigned mid = moduleIds[j];
            grad_[mid].x += (exp_x[j] / sumExpX - exp_neg_x[j] / sumExpNegX);
            grad_[mid].y += (exp_y[j] / sumExpY - exp_neg_y[j] / sumExpNegY);
        }
    }

    return grad_;
}
