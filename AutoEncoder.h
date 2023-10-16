#include <iostream>
#include <C:\eigen-3.4.0\Eigen\Dense>
#include <vector>
#include "Layer.h"
#include "Activation.h"
class AutoEncoder {
    private:
    Layer layer1;
    Layer layer2;
    Layer layer3;
    Eigen::VectorXd z1;
    Eigen::VectorXd z2;
    Eigen::VectorXd z3;
    Eigen::VectorXd target;
    Eigen::VectorXd error;
    public: 
        AutoEncoder();
        void train(Eigen::MatrixXd m, std::vector<double> labels, double learningRate, int epochs);
};