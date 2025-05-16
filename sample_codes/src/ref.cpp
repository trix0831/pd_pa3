#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include "Placement.h"
#include "Point.h"
#include "TrueObj.h"  // <-- corrected to match your actual header filename
#include "Net.h"
#include "Module.h"
#include "Pin.h"

#include <cstdio>

trueObj::trueObj(Placement &placement, double lambda)
    : placement_(placement),
      wirelength_value_(0.0),
      density_value_(0.0),
      lambda_(lambda),
      op_val_(0.0)
{
    /* temp buckets: one vector<Pin*> per net */
    std::vector<std::vector<Pin*>> pinBuckets(placement_.numNets());

    /* pass 1: bucket every pin by its netId */
    for (unsigned pid = 0; pid < placement_.numPins(); ++pid) {
        Pin &p = placement_.pin(pid);
        int nid = p.netId();
        if (nid < 0 || (unsigned)nid >= placement_.numNets()) continue;
        pinBuckets[nid].push_back(&p);
    }

    /* pass 2: replace each net’s pin list + correct _numPins */
    for (unsigned nid = 0; nid < placement_.numNets(); ++nid) {
        Net &net = placement_.net(nid);

        const unsigned real = pinBuckets[nid].size();
        net.clearPins();              // vector empty, _numPins still old
        net.setNumPins(real);         // now _numPins = real and vector has
                                      // 'real' NULL placeholders
        for (unsigned i = 0; i < real; ++i)
            net._pPins[i] = pinBuckets[nid][i];   // overwrite placeholders
    }

#ifdef DEBUG_BUILD
    printf("[trueObj] rebuilt nets: nNets=%u  pin0=%u  pin1=%u\n",
           placement_.numNets(),
           placement_.net(0).numPins(),
           placement_.net(1).numPins());
#endif

    /* gradient buffers */
    size_t nMod = placement_.numModules();
    wirelength_gradient_.assign(nMod, Point2<double>(0,0));
    density_gradient_.assign(nMod, Point2<double>(0,0));
}




double trueObj::HandleWirelength(std::vector<Point2<double>> &input)
{
    const double alpha = 1.0;                      // LSE smooth factor
    const size_t numMods = placement_.numModules();

    wirelength_value_     = 0.0;
    wirelength_gradient_.assign(numMods, Point2<double>(0.0, 0.0));

    /* ---------- iterate over all nets ---------- */
    printf(">>> trueObj::HandleWirelength: numNets=%u\n",
           placement_.numNets());
    for (unsigned netId = 0; netId < placement_.numNets(); ++netId) {
        Net &net = placement_.net(netId);
        const size_t np = net.numPins();
        if (np <= 1) {
            printf("net %u: npins=%zu, skip\n", netId, np);
            continue;                               // skip trivial nets
        }

        /* collect pin coordinates */
        std::vector<double> xs(np), ys(np);
        std::vector<int>    movableId(np);         // −1 ⇒ fixed pin

        double max_x = -std::numeric_limits<double>::infinity();
        double max_y = -std::numeric_limits<double>::infinity();

        printf("net %u: npins=%zu\n", netId, np);
        for (size_t k = 0; k < np; ++k) {
            Pin &pin = net.pin(k);
            unsigned m = pin.moduleId();

            if (m < numMods) {                     // movable module
                xs[k] = input[m].x;
                ys[k] = input[m].y;
                movableId[k] = static_cast<int>(m);
            } else {                               // fixed terminal / pad
                xs[k] = pin.x();                   // use absolute pin coord
                ys[k] = pin.y();
                movableId[k] = -1;                 // mark as fixed
            }
            if (xs[k] > max_x) max_x = xs[k];
            if (ys[k] > max_y) max_y = ys[k];
        }

        printf("net %u: max_x=%.3f max_y=%.3f\n", netId, max_x, max_y);

        /* log-sum-exp wire-length */
        double sx = 0.0, sNx = 0.0, sy = 0.0, sNy = 0.0;
        for (size_t k = 0; k < np; ++k) {
            sx  += std::exp(alpha * (xs[k] - max_x));
            sNx += std::exp(alpha * (max_x - xs[k]));
            sy  += std::exp(alpha * (ys[k] - max_y));
            sNy += std::exp(alpha * (max_y - ys[k]));
        }

        printf("neeeet %u: max_x=%.3f max_y=%.3f\n", netId, max_x, max_y);

        double wl_x = (std::log(sx)  + max_x + std::log(sNx) - max_x) / alpha;
        double wl_y = (std::log(sy)  + max_y + std::log(sNy) - max_y) / alpha;
        printf("net %u: wl_x=%.3f wl_y=%.3f\n", netId, wl_x, wl_y);
        wirelength_value_ += wl_x + wl_y;

        /* gradient for movable modules only */
        for (size_t k = 0; k < np; ++k) {
            const int m = movableId[k];
            if (m < 0) continue;                  // skip fixed pins

            double gx = alpha * std::exp(alpha * (xs[k] - max_x)) / sx
                      - alpha * std::exp(alpha * (max_x - xs[k])) / sNx;
            double gy = alpha * std::exp(alpha * (ys[k] - max_y)) / sy
                      - alpha * std::exp(alpha * (max_y - ys[k])) / sNy;

            wirelength_gradient_[m].x += gx;
            wirelength_gradient_[m].y += gy;
        }
    }
    return wirelength_value_;
}



// Bell kernel φ(d) = 1 − (d/R)^2 if |d| < R, else 0
static inline double bell(double d, double R) {
    if (std::fabs(d) >= R) return 0.0;
    double t = d / R;
    return 1.0 - t * t;
}

// Derivative dφ/dd = −2d/R² if |d| < R, else 0
static inline double bell_grad(double d, double R) {
    return (std::fabs(d) >= R) ? 0.0 : -2.0 * d / (R * R);
}

double trueObj::HandleDensity(std::vector<Point2<double>> &input) {
    // Chip boundary
    double lx = placement_.boundryLeft();
    double ly = placement_.boundryBottom();
    double ux = placement_.boundryRight();
    double uy = placement_.boundryTop();

    // Bin grid size
    const unsigned nx = 32, ny = 32;
    const double bw = (ux - lx) / nx;
    const double bh = (uy - ly) / ny;
    const double Rx = 2.0 * bw;
    const double Ry = 2.0 * bh;
    const double R_area = Rx * Ry;
    const double cap = bw * bh * 0.9;

    const size_t numMods = placement_.numModules();
    density_value_ = 0.0;
    density_gradient_.assign(numMods, Point2<double>(0.0, 0.0));

    struct Bin { double rho = 0.0; };
    std::vector<Bin> bins(nx * ny);

    // 1. Accumulate density per bin
    for (unsigned m = 0; m < numMods; ++m) {
        Module& mod = placement_.module(m);
        double x = input[m].x;
        double y = input[m].y;
        double area = mod.area();

        int ix_min = std::max(0, static_cast<int>((x - Rx - lx) / bw));
        int ix_max = std::min((int)nx - 1, static_cast<int>((x + Rx - lx) / bw));
        int iy_min = std::max(0, static_cast<int>((y - Ry - ly) / bh));
        int iy_max = std::min((int)ny - 1, static_cast<int>((y + Ry - ly) / bh));

        for (int ix = ix_min; ix <= ix_max; ++ix) {
            double cx = lx + (ix + 0.5) * bw;
            double φx = bell(x - cx, Rx);
            if (φx == 0.0) continue;

            for (int iy = iy_min; iy <= iy_max; ++iy) {
                double cy = ly + (iy + 0.5) * bh;
                double φy = bell(y - cy, Ry);
                if (φy == 0.0) continue;

                bins[iy * nx + ix].rho += (area * φx * φy) / R_area;
            }
        }
    }

    // 2. Compute penalty and gradients
    for (unsigned ix = 0; ix < nx; ++ix) {
        for (unsigned iy = 0; iy < ny; ++iy) {
            double rho = bins[iy * nx + ix].rho;
            double overflow = rho - cap;
            if (overflow <= 0.0) continue;

            double cx = lx + (ix + 0.5) * bw;
            double cy = ly + (iy + 0.5) * bh;
            double coeff = lambda_ * overflow;

            for (unsigned m = 0; m < numMods; ++m) {
                Module& mod = placement_.module(m);
                double x = input[m].x;
                double y = input[m].y;
                double area = mod.area();

                double dx = x - cx;
                double dy = y - cy;
                double φx = bell(dx, Rx);
                double φy = bell(dy, Ry);
                if (φx == 0.0 || φy == 0.0) continue;

                double dφx = bell_grad(dx, Rx);
                double dφy = bell_grad(dy, Ry);
                double aR = area / R_area;

                density_gradient_[m].x += coeff * aR * dφx * φy;
                density_gradient_[m].y += coeff * aR * φx * dφy;
            }

            density_value_ += 0.5 * lambda_ * overflow * overflow;
        }
    }

    return density_value_;
}


double trueObj::HandleUpdate(std::vector<Point2<double>> &input) {
    printf("Wirelength: %f\n", HandleWirelength(input));
    printf("Density: %f\n", HandleDensity(input));
    
    return 0.0;  // Placeholder return
}
