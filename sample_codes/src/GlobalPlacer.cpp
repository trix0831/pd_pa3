#include "GlobalPlacer.h"

#include <cstdio>
#include <vector>
#include <set>

#include "ObjectiveFunction.h"
#include "Optimizer.h"
#include "Point.h"

GlobalPlacer::GlobalPlacer(Placement &placement)
    : _placement(placement) {
}

void GlobalPlacer::place() {
    ////////////////////////////////////////////////////////////////////
    // This section is an example for analytical methods.
    // The objective is to minimize the following function:
    //      f(x,y) = 3*x^2 + 2*x*y + 2*y^2 + 7
    //
    // If you use other methods, you can skip and delete it directly.
    ////////////////////////////////////////////////////////////////////
    std::vector<Point2<double>> tempPos(_placement.numModules());       
    _placement.setWidth();
    _placement.setHeight();
    // randomly initialize the position of the modules
    srand(time(0));
    double outline_w = abs(_placement.boundryRight() - _placement.boundryLeft());
    double outline_h = abs(_placement.boundryTop() - _placement.boundryBottom());
    // cout << "outline: (" << _placement.boundryLeft() << "," << _placement.boundryBottom() << ") (" << _placement.boundryRight() << "," << _placement.boundryTop() << ")\n";
    // cout << "outline_w=" << outline_w << ", outline_h=" << outline_w << "\n";
    for( size_t i = 0; i < _placement.numModules(); ++i ){
        // Module &module = _placement.module(i);
        // int x = rand() % int(outline_w) + _placement.boundryLeft();
        // int y = rand() % int(outline_h) + _placement.boundryBottom();
        // int x = (rand() % int(outline_w))*2/3 + _placement.boundryLeft() + int(outline_w)/6;
        // int y = (rand() % int(outline_h))*2/3 + _placement.boundryBottom() + int(outline_h)/6;
        int x = _placement.boundryLeft() + int(outline_w)/2;
        int y = _placement.boundryBottom() + int(outline_h)/2;
        // module.setPosition(x, y);
        tempPos[i].x = x;
        tempPos[i].y = y;
        // if(i % 1000 == 0)
        //     cout << "module[" << i << "]: (" << x << "," << y << ")\n";
    }
    
    // // Optimization variables (in this example, there is only one tempPos)
    ExampleFunction foo(_placement, 1);       // Objective function, lambda start with 1
    // const double kAlpha = 0.00000000003;                         // Constant step size
    const double kAlpha = 0.000000003;
    SimpleConjugateGradient optimizer(foo, tempPos, kAlpha, _placement);  // Optimizer

    // // Set initial point
    // tempPos[0] = 4.;  // This set both tempPos[0].x and tempPos[0].y to 4.

    // Initialize the optimizer
    optimizer.Initialize();

    // Perform optimization, the termination condition is that the number of iterations reaches 100
    // TODO: You may need to change the termination condition, which is determined by the overflow ratio.
    double spread = 0;
    int i = 0;
    while(spread < 0.2) {
        spread = optimizer.Step();
        printf("iter = %3u, f = %9.4f, x = %9.4f, y = %9.4f\n", i, foo(tempPos), tempPos[0].x, tempPos[0].y);
        i++;
    }


    // ////////////////////////////////////////////////////////////////////
    // // Global placement algorithm
    // ////////////////////////////////////////////////////////////////////

    // TODO: Implement your global placement algorithm here.
    const size_t num_modules = _placement.numModules();  // You may modify this line.
    // std::vector<Point2<double>> positions(num_modules);  // Optimization variables (positions of modules). You may modify this line.

    ////////////////////////////////////////////////////////////////////
    // Write the placement result into the database. (You may modify this part.)
    ////////////////////////////////////////////////////////////////////
    // size_t fixed_cnt = 0;
    for (size_t i = 0; i < num_modules; i++) {
        // If the module is fixed, its position should not be changed.
        // In this programing assignment, a fixed module may be a terminal or a pre-placed module.
        // if (_placement.module(i).isFixed()) {
        //     fixed_cnt++;
        //     continue;
        // }
        _placement.module(i).setPosition(tempPos[i].x, tempPos[i].y);
    }
    // printf("INFO: %lu / %lu modules are fixed.\n", fixed_cnt, num_modules);
}

void GlobalPlacer::plotPlacementResult(const string outfilename, bool isPrompt) {
    ofstream outfile(outfilename.c_str(), ios::out);
    outfile << " " << endl;
    outfile << "set title \"wirelength = " << _placement.computeHpwl() << "\"" << endl;
    outfile << "set size ratio 1" << endl;
    outfile << "set nokey" << endl
            << endl;
    outfile << "plot[:][:] '-' w l lt 3 lw 2, '-' w l lt 1" << endl
            << endl;
    outfile << "# bounding box" << endl;
    plotBoxPLT(outfile, _placement.boundryLeft(), _placement.boundryBottom(), _placement.boundryRight(), _placement.boundryTop());
    outfile << "EOF" << endl;
    outfile << "# modules" << endl
            << "0.00, 0.00" << endl
            << endl;
    for (size_t i = 0; i < _placement.numModules(); ++i) {
        Module &module = _placement.module(i);
        plotBoxPLT(outfile, module.x(), module.y(), module.x() + module.width(), module.y() + module.height());
    }
    outfile << "EOF" << endl;
    outfile << "pause -1 'Press any key to close.'" << endl;
    outfile.close();

    if (isPrompt) {
        char cmd[200];
        sprintf(cmd, "gnuplot %s", outfilename.c_str());
        if (!system(cmd)) {
            cout << "Fail to execute: \"" << cmd << "\"." << endl;
        }
    }
}

void GlobalPlacer::plotBoxPLT(ofstream &stream, double x1, double y1, double x2, double y2) {
    stream << x1 << ", " << y1 << endl
           << x2 << ", " << y1 << endl
           << x2 << ", " << y2 << endl
           << x1 << ", " << y2 << endl
           << x1 << ", " << y1 << endl
           << endl;
}
