#include <iostream>
#include <C:\eigen-3.4.0\Eigen\Dense>
#include <vector>
#include <omp.h>
#include "Layer.h"
class AutoEncoder {
    private:
    Layer layers;
    Eigen::VectorXd target;
    std::vector<int> layerSizes;
    double learningRate;
    int epochs;
    int batchSize;
    public: 
        AutoEncoder(int numLayers, int sizes...);
        void train(Eigen::MatrixXd m, std::vector<double> labels, double lr, int epochs, int batchSize);
};