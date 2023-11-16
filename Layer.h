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
        std::vector<Eigen::MatrixXd> gradWeights;
        std::vector<Eigen::VectorXd> gradBiases;
        std::vector<int> layerSizes;
        Eigen::VectorXd delta;
        Eigen::VectorXd err;
        double beta1, beta2, epsilon;
        Eigen::MatrixXd mtW, vtW;
        Eigen::VectorXd mtB, vtB;
        Eigen::MatrixXd mtW_corr, vtW_corr;
        Eigen::VectorXd mtB_corr, vtB_corr;
        int t;
        double tot_err;
    public:
        //template <typename... Args>
        Layer();
        Layer(int numLayers, std::vector<int> inputLayers);
        void feedForward(Eigen::VectorXd input, Eigen::VectorXd target);
        void backwardsPropagation(Eigen::VectorXd input, double lr);
        void updateWeightsBiases(double lr);
        void resetTotErr();
        double getTotErr();
};