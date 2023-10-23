#include <iostream>
#include <C:\eigen-3.4.0\Eigen\Dense>
#include "Activation.h"
class Layer {
    private:
        Eigen::MatrixXd weight;
        Eigen::VectorXd bias;
        Eigen::VectorXd z;
    public:
        Layer();
        Layer(int row, int col);
        Eigen::MatrixXd getWeight();
        Eigen::VectorXd getBias();
        Eigen::VectorXd getZ();
        void updateWeights(Eigen::MatrixXd newWeight);
        void updateBiases(Eigen::VectorXd newBias);
        void feedForward(Eigen::VectorXd input, std::string activate);

};