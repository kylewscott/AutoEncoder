#include "Layer.h"
//Constructor
Layer::Layer(){}
Layer::Layer(int numLayers, std::vector<int> inputLayers) {
    //Layer sizes given from user
    layerSizes = inputLayers;
    //initialize weights, biases, and gradients
    for(size_t i = 0; i < layerSizes.size()-1; i++){
        weights.push_back(Eigen::MatrixXd::Random(layerSizes[i+1], layerSizes[i]) * std::sqrt(2.0 / layerSizes[i+1]));
        biases.push_back(Eigen::VectorXd::Random(layerSizes[i+1]) * std::sqrt(2.0 / layerSizes[i+1]));
        gradWeights.push_back(Eigen::MatrixXd::Zero(1,1));
        gradBiases.push_back(Eigen::VectorXd::Zero(1));
    }
    //initialize layers with number of layers given
    layers = std::vector<Eigen::VectorXd>(layerSizes.size());
    //initialize adam values
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = std::exp(-8);
    t = 0;
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
    //input layer
    layers[0] = input;
    //hidden layers
    for(size_t i = 1; i < layers.size()-1; i ++){
        layers[i] = ((weights[i-1] * layers[i-1]) + biases[i-1]);
        a.relu(layers[i]);
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
    //find gradients
    delta = a.softMaxPrime(layers[layers.size()-1]) * err;
    gradWeights[gradWeights.size()-1] = (delta * layers[layers.size()-2].transpose());
    gradBiases[gradBiases.size()-1] = delta;
    for(size_t i = gradWeights.size()-1 ; i-- > 0 ;){
        delta = (weights[i+1].transpose() * delta).array() * a.reluPrime(layers[i+1]).array();
        gradWeights[i] = (delta * layers[i].transpose());
        gradBiases[i] = delta;
    }
}
//update weights and biases
void Layer::updateWeightsBiases(double lr){
    //Adam
    for(size_t i = weights.size() ; i-- > 0 ;){
        t++;
        //intialize movements
        mtW = Eigen::MatrixXd::Zero(gradWeights[i].rows(), gradWeights[i].cols());
        mtB = Eigen::VectorXd::Zero(gradBiases[i].size());
        vtW = Eigen::MatrixXd::Zero(gradWeights[i].rows(), gradWeights[i].cols());
        vtB = Eigen::VectorXd::Zero(gradBiases[i].size());
        //Set Movements
        mtW = beta1 * mtW + (1 - beta1) * gradWeights[i];
        mtB = beta1 * mtB + (1 - beta1) * gradBiases[i];
        vtW = beta2 * vtW.array() + (1 - beta2) * gradWeights[i].array().square();
        vtB = beta2 * vtB.array() + (1 - beta2) * gradBiases[i].array().square();
        //Make corrections
        mtW_corr = mtW.array() / (1-std::pow(beta1, t));
        mtB_corr = mtB.array() / (1-std::pow(beta1, t));
        vtW_corr = vtW.array() / (1-std::pow(beta2, t));
        vtB_corr = vtB.array() / (1-std::pow(beta2, t));
        //update the weights and biases;
        weights[i] = weights[i].array() + (lr * (mtW_corr.array() / (vtW_corr.array().sqrt() + epsilon)));
        biases[i] = biases[i].array() + (lr * (mtB_corr.array() / (vtB_corr.array().sqrt() + epsilon)));
        }

}