#include "Layer.h"
//Constructor
Layer::Layer(){}
Layer::Layer(int sizes...){
    //Loads sizes into the layerSizes vector
    std::va_list args;
    va_start(args, sizes);
    layerSizes.push_back(sizes);
    for(int i = 0; i < sizeof(sizes)-1; ++i){
        layerSizes.push_back(va_arg(args, int));
    }
    va_end(args);
    //initialize weights and biases
    for(size_t i = 0; i < layerSizes.size()-1; i++){
        weights.push_back(Eigen::MatrixXd::Random(layerSizes[i+1], layerSizes[i]) * std::sqrt(6.0 / (layerSizes[i] + layerSizes[layerSizes.size()-1])));
        biases.push_back(Eigen::VectorXd::Random(layerSizes[i+1]));
    }
    //initialize layers with number of layers given
    layers = std::vector<Eigen::VectorXd>(layerSizes.size());
}
//Reset the total error
void Layer::resetTotErr(){
    tot_err = 0;
}
//Return total error
double Layer::getTotErr(){
    return tot_err;
}
//Forward propagation
void Layer::feedForward(Eigen::VectorXd input, Eigen::VectorXd target){
    Activation a;
    //input
    layers[0] = input;
    //hidden layers
    for(size_t i = 1; i < layers.size()-1; i ++){
        layers[i] = ((weights[i-1] * layers[i-1]) + biases[i-1]);
        a.sigmoid(layers[i]);
    }
    //output layer
    layers[layers.size()-1] = ((weights[layers.size()-2] * layers[layers.size()-2]) + biases[layers.size()-2]);
    a.softMax(layers[layers.size()-1]);
    //Calculate error
    err = target - layers[layers.size()-1];
    err.array() /= 10;
    tot_err += err.array().abs().sum();
}
//Backwards propagation
void Layer::backwardsPropagation(Eigen::VectorXd input, double lr){
    Activation a;
    //Output layer updates
    delta = a.softMaxPrime(layers[layers.size()-1]) * err;
    weights[weights.size()-1] = weights[weights.size()-1].array() + (lr * (delta * layers[layers.size()-2].transpose())).array();
    biases[biases.size()-1] = biases[biases.size()-1] + (lr * delta);
    //all other layers updates
    for(size_t i = weights.size()-1 ; i-- > 0 ;){
        delta = (weights[i+1].transpose() * delta).array() * a.sigmoidPrime(layers[i+1]).array();
        weights[i] = weights[i].array() + (lr * (delta * layers[i].transpose())).array();
        biases[i] = biases[i] + (lr * delta);
    }
}
