#include <iostream>
#include <cstdarg>
#include <vector>
#include <C:\eigen-3.4.0\Eigen\Dense>
#include "Activation.h"
class Layer {
    private:
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::VectorXd> biases;
        std::vector<Eigen::VectorXd> layers;
        std::vector<int> layerSizes;
        Eigen::VectorXd delta;
        Eigen::VectorXd err;
        double tot_err;
    public:
        //template <typename... Args>
        Layer();
        Layer(int sizes...);
        void feedForward(Eigen::VectorXd input, Eigen::VectorXd target);
        void backwardsPropagation(Eigen::VectorXd input, double lr);
        void resetTotErr();
        double getTotErr();
};