#include "ObjectiveFunction.h"

#include "cstdio"

/* ---------- tunables ---------------------------------------------------- */
static constexpr int    kBinX   = 64;   // number of bins in X
static constexpr int    kBinY   = 64;   // number of bins in Y
static constexpr double kTarget = 0.85; // target utilisation per bin
static constexpr double kWeight = 1e3;  // λd – weight of the density term
/* ------------------------------------------------------------------------ */

static inline double bell(double u)
/* raised-cosine bell:  0 … |u|≥1 ,  (1+cos(πu))/2 … |u|<1                        */
{
    u = std::fabs(u);
    return (u >= 1.0) ? 0.0 : 0.5 * (1.0 + std::cos(M_PI * u));
}

ExampleFunction::ExampleFunction(Placement &placement) : BaseFunction(1), placement_(placement)
{
    printf("Fetch the information you need from placement database.\n");
    printf("For example:\n");
    printf("    Placement boundary: (%.f,%.f)-(%.f,%.f)\n", placement_.boundryLeft(), placement_.boundryBottom(),
           placement_.boundryRight(), placement_.boundryTop());
}



const double &ExampleFunction::operator()(const std::vector<Point2<double>> &input)
{
    /* ============ 1.  wire-length (your existing code) =================== */
    value_ = 0.0;
    double value_wire = 0.0;
    double value_density = 0.0;
    input_ = input;
    const double alpha = 1.0;

    for (unsigned i = 0; i < placement_.numNets(); ++i)
    {
        Net &net = placement_.net(i);
        std::vector<double> xs, ys;

        for (unsigned j = 0; j < net.numPins(); ++j)
        {
            const Point2<double> &pt = input[net.pin(j).moduleId()];
            xs.push_back(pt.x);
            ys.push_back(pt.y);
        }

        const double xmax = *std::max_element(xs.begin(), xs.end());
        const double xmin = *std::min_element(xs.begin(), xs.end());
        const double ymax = *std::max_element(ys.begin(), ys.end());
        const double ymin = *std::min_element(ys.begin(), ys.end());

        double sx = 0, sNx = 0, sy = 0, sNy = 0;
        for (std::size_t k = 0; k < xs.size(); ++k)
        {
            sx  += std::exp((xs[k] - xmax) / alpha);
            sNx += std::exp((xmin - xs[k]) / alpha);
            sy  += std::exp((ys[k] - ymax) / alpha);
            sNy += std::exp((ymin - ys[k]) / alpha);
        }

        value_wire += alpha * (std::log(sx) + xmax / alpha + std::log(sNx) + xmin / alpha);
        value_wire += alpha * (std::log(sy) + ymax / alpha + std::log(sNy) + ymin / alpha);
    }

    /* ============ 2.  density with bell-shaped kernel ==================== */

    /* --- 2-a. grid and helpers ------------------------------------------ */
    const double outlineL = placement_.boundryLeft();
    const double outlineR = placement_.boundryRight();
    const double outlineB = placement_.boundryBottom();
    const double outlineT = placement_.boundryTop();

    const double binW = (outlineR - outlineL) / kBinX;
    const double binH = (outlineT - outlineB) / kBinY;
    const double binCapacity = binW * binH * kTarget;

    /* ρ[i][j] – accumulated area in bin (i,j) */
    static double rho[kBinX][kBinY];
    for (int i = 0; i < kBinX; ++i)
        for (int j = 0; j < kBinY; ++j) rho[i][j] = 0.0;

    /* --- 2-b. splat every module onto nearby bins ------------------------ */
    const double supportX = 2.0 * binW;   // bell influence = 2 bins in each dir.
    const double supportY = 2.0 * binH;

    for (unsigned m = 0; m < placement_.numModules(); ++m)
    {
        Module &mod = placement_.module(m);
        const double cx = input[m].x + 0.5 * mod.width();   // centre of module
        const double cy = input[m].y + 0.5 * mod.height();
        const double area = mod.area();

        /* bins overlapped by the kernel support --------------------------- */
        const int iLo = std::max(0, int((cx - supportX - outlineL) / binW));
        const int iHi = std::min(kBinX - 1, int((cx + supportX - outlineL) / binW));
        const int jLo = std::max(0, int((cy - supportY - outlineB) / binH));
        const int jHi = std::min(kBinY - 1, int((cy + supportY - outlineB) / binH));

        for (int i = iLo; i <= iHi; ++i)
        {
            const double binCenterX = outlineL + (i + 0.5) * binW;
            const double dx = (cx - binCenterX) / supportX; // normalised dist.
            const double fx = bell(dx);

            if (fx == 0.0) continue;

            for (int j = jLo; j <= jHi; ++j)
            {
                const double binCenterY = outlineB + (j + 0.5) * binH;
                const double dy = (cy - binCenterY) / supportY;
                const double fy = bell(dy);

                if (fy == 0.0) continue;

                /* contribution (separable kernel):  A * fx * fy ------------ */
                rho[i][j] += area * fx * fy;
            }
        }
    }

    /* --- 2-c. overflow energy ------------------------------------------- */
    double densityCost = 0.0;
    for (int i = 0; i < kBinX; ++i)
        for (int j = 0; j < kBinY; ++j)
        {
            // const double overflow = std::max(0.0, rho[i][j] - binCapacity);
            // densityCost += overflow * overflow;
            const double diff = rho[i][j] - binCapacity;     // can be ±
            densityCost += diff * diff;
        }

    value_density += kWeight * densityCost;

    printf("Wirelength: %.3f\n", value_wire);
    printf("Density: %.3f\n", value_density);
    value_ = value_wire + value_density;
    printf("Total: %.3f\n", value_);
    return value_;
}

const std::vector<Point2<double>> &ExampleFunction::Backward()
{
    /* ------------------------------------------------------------------ *
     * 0.  initialise gradient                                             *
     * ------------------------------------------------------------------ */
    grad_.assign(input_.size(), Point2<double>(0.0, 0.0));
    std::vector<Point2<double>> grad_wire_, grad_density_;
    grad_wire_.assign(input_.size(), Point2<double>(0.0, 0.0));
    grad_density_.assign(input_.size(), Point2<double>(0.0, 0.0));

    /* ================================================================== *
     * 1.  WIRE-LENGTH  (your original code, unchanged)                   *
     * ================================================================== */
    const double alpha_wl = 0.01;               //   same value you used
    for (unsigned i = 0; i < placement_.numNets(); ++i)
    {
        Net &net = placement_.net(i);
        const unsigned npins = net.numPins();

        std::vector<unsigned> mids(npins);
        std::vector<double>   xs(npins), ys(npins);

        for (unsigned j = 0; j < npins; ++j)
        {
            Pin &pin   = net.pin(j);
            mids[j]    = pin.moduleId();
            xs[j]      = input_[mids[j]].x;
            ys[j]      = input_[mids[j]].y;
        }

        /* ----- x ------------------------------------------------------- */
        const double xmax = *std::max_element(xs.begin(), xs.end());
        const double xmin = *std::min_element(xs.begin(), xs.end());

        std::vector<double> ex(npins), enx(npins);
        double sumEx = 0.0, sumEnx = 0.0;
        for (unsigned j = 0; j < npins; ++j)
        {
            ex [j] = std::exp((xs[j] - xmax) / alpha_wl);
            enx[j] = std::exp((xmin - xs[j]) / alpha_wl);
            sumEx  += ex [j];
            sumEnx += enx[j];
        }

        /* ----- y ------------------------------------------------------- */
        const double ymax = *std::max_element(ys.begin(), ys.end());
        const double ymin = *std::min_element(ys.begin(), ys.end());

        std::vector<double> ey(npins), eny(npins);
        double sumEy = 0.0, sumEny = 0.0;
        for (unsigned j = 0; j < npins; ++j)
        {
            ey [j] = std::exp((ys[j] - ymax) / alpha_wl);
            eny[j] = std::exp((ymin - ys[j]) / alpha_wl);
            sumEy  += ey [j];
            sumEny += eny[j];
        }

        /* ----- accumulate module-wise gradient ------------------------- */
        for (unsigned j = 0; j < npins; ++j)
        {
            const unsigned mid = mids[j];
            grad_wire_[mid].x +=  (ex [j] / sumEx)  - (enx[j] / sumEnx);
            grad_wire_[mid].y +=  (ey [j] / sumEy)  - (eny[j] / sumEny);
        }
    }

    /* ================================================================== *
     * 2.  DENSITY  (bell-shaped kernel)                                  *
     * ================================================================== */

    /* ---- 2-a.  constants ­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­ */
    constexpr int    kBinX   = 64;
    constexpr int    kBinY   = 64;
    constexpr double kTarget = 0.85;    // target bin utilisation
    constexpr double kWeight = 1e3;     // λd  – weight in the cost

    const double L = placement_.boundryLeft();
    const double R = placement_.boundryRight();
    const double B = placement_.boundryBottom();
    const double T = placement_.boundryTop();

    const double binW = (R - L) / kBinX;
    const double binH = (T - B) / kBinY;
    const double cap  = binW * binH * kTarget;

    const double supX = 2.0 * binW;     // kernel reach  = 2 bins
    const double supY = 2.0 * binH;

    /* ---- 2-b.  helper lambdas (bell and its derivative) -------------- */
    auto bell = [](double u) -> double
    {
        u = std::fabs(u);
        return (u >= 1.0) ? 0.0 : 0.5 * (1.0 + std::cos(M_PI * u));
    };
    auto dbell = [](double u) -> double          // d/du of bell(u)
    {
        if (std::fabs(u) >= 1.0) return 0.0;
        return -0.5 * M_PI * std::sin(M_PI * u);
    };

    /* ---- 2-c.  first pass – accumulate area per bin ------------------- */
    std::vector<std::vector<double>> rho(kBinX, std::vector<double>(kBinY, 0.0));

    for (unsigned m = 0; m < placement_.numModules(); ++m)
    {
        Module &mod = placement_.module(m);
        const double cx   = input_[m].x + 0.5 * mod.width();
        const double cy   = input_[m].y + 0.5 * mod.height();
        const double area = mod.area();

        const int iLo = std::max(0, int((cx - supX - L) / binW));
        const int iHi = std::min(kBinX - 1, int((cx + supX - L) / binW));
        const int jLo = std::max(0, int((cy - supY - B) / binH));
        const int jHi = std::min(kBinY - 1, int((cy + supY - B) / binH));

        for (int i = iLo; i <= iHi; ++i)
        {
            const double bcX = L + (i + 0.5) * binW;
            const double ux  = (cx - bcX) / supX;     // normalised distance
            const double fx  = bell(ux);
            if (fx == 0.0) continue;

            for (int j = jLo; j <= jHi; ++j)
            {
                const double bcY = B + (j + 0.5) * binH;
                const double uy  = (cy - bcY) / supY;
                const double fy  = bell(uy);
                if (fy == 0.0) continue;

                rho[i][j] += area * fx * fy;
            }
        }
    }

    /* ---- 2-d.  pre-compute overflow Δ_{ij} ---------------------------- */
    std::vector<std::vector<double>> ov(kBinX, std::vector<double>(kBinY, 0.0));
    for (int i = 0; i < kBinX; ++i)
        for (int j = 0; j < kBinY; ++j)
            ov[i][j] = std::max(0.0, rho[i][j] - cap);
const double binCapacity = binW * binH * kTarget;
    /* ---- 2-e.  second pass – accumulate gradients --------------------- */
    for (unsigned m = 0; m < placement_.numModules(); ++m)
    {
        Module &mod   = placement_.module(m);
        const double cx   = input_[m].x + 0.5 * mod.width();
        const double cy   = input_[m].y + 0.5 * mod.height();
        const double area = mod.area();

        const int iLo = std::max(0, int((cx - supX - L) / binW));
        const int iHi = std::min(kBinX - 1, int((cx + supX - L) / binW));
        const int jLo = std::max(0, int((cy - supY - B) / binH));
        const int jHi = std::min(kBinY - 1, int((cy + supY - B) / binH));

        double gx = 0.0, gy = 0.0;

        for (int i = iLo; i <= iHi; ++i)
        {
            const double bcX = L + (i + 0.5) * binW;
            const double ux  = (cx - bcX) / supX;
            const double fx  = bell (ux);
            const double dfx = dbell(ux) / supX;   // ∂fx/∂x

            if (fx == 0.0 && dfx == 0.0) continue;

            for (int j = jLo; j <= jHi; ++j)
            {
                // const double delta = ov[i][j];
                // if (delta <= 0.0) continue;        // no overflow ⇒ no force
                const double delta = rho[i][j] - binCapacity;   // signed deviation
                if (std::fabs(delta) < 1e-12) continue; 

                const double bcY = B + (j + 0.5) * binH;
                const double uy  = (cy - bcY) / supY;
                const double fy  = bell (uy);
                const double dfy = dbell(uy) / supY; // ∂fy/∂y

                if (fy == 0.0 && dfy == 0.0) continue;

                const double coef = 2.0 * kWeight * delta * area; // d(Δ²)/dΔ

                gx += coef * dfx * fy;   // ∂ρ/∂x = area·dfx·fy
                gy += coef * fx  * dfy;  // ∂ρ/∂y = area·fx ·dfy
            }
        }

        grad_density_[m].x += gx;
        grad_density_[m].y += gy;
    }

    //print gradient of density and wirelength
    printf("grad_density x: %f", grad_density_[0].x);
    printf(" grad_density y: %f\n", grad_density_[0].y);
    printf("grad_wire x: %f", grad_wire_[0].x);
    printf(" grad_wire y: %f\n", grad_wire_[0].y);
    /* ---- 2-f.  combine gradients ------------------------------------- */
    for (unsigned m = 0; m < placement_.numModules(); ++m)
    {
        grad_[m].x = grad_wire_[m].x + grad_density_[m].x;
        grad_[m].y = grad_wire_[m].y + grad_density_[m].y;
    }


    return grad_;
}
