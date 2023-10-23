#include "Layer.h"
//Constructors
Layer::Layer(){};
Layer::Layer(int row, int col) {
    //initialize weights and biases for layer
    weight = Eigen::MatrixXd::Random(row, col) * std::sqrt(6.0 / (col + 10));
    //weight = weight.array().abs();
    bias = Eigen::VectorXd::Zero(row);
    //bias = bias.array().abs();
}
//Getters
Eigen::MatrixXd Layer::getWeight() {
    return weight;
}
Eigen::VectorXd Layer::getBias() {
    return bias;
}
Eigen::VectorXd Layer::getZ() {
    return z;
}
void Layer::feedForward(Eigen::VectorXd input, std::string activate) {
    Activation a;
    if(activate == "sigmoid"){
        z = (weight * input) + bias;
        a.sigmoid(z);
    }
    if(activate == "softmax"){
        z = (weight * input) + bias;
        a.softMax(z);
    }
}
//Update functions
void Layer::updateWeights(Eigen::MatrixXd newWeight) {
    weight = newWeight;
}
void Layer::updateBiases(Eigen::VectorXd newBias) {
    bias = newBias;
}

