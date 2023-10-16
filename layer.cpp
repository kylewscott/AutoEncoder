#include "Layer.h"
//Constructors
Layer::Layer(){};
Layer::Layer(int row, int col) {
    //initialize weights and biases for layer
    weight = Eigen::MatrixXd::Random(row, col);
    weight = weight.array().abs();
    bias = Eigen::VectorXd::Random(row);
}
//Getters
Eigen::MatrixXd Layer::getWeight() {
    return weight;
}
Eigen::VectorXd Layer::getBias() {
    return bias;
}
//Update functions
void Layer::updateWeights(Eigen::MatrixXd newWeight) {
    weight = newWeight;
}
void Layer::updateBiases(Eigen::VectorXd newBias) {
    bias = newBias;
}
