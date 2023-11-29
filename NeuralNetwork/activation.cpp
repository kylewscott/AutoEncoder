#include "Activation.h"
//Activation Functions and Derivatives
void Activation::sigmoid(Eigen::VectorXd& x) {
    x = 1.0/(1.0 + (-x.array()).exp());
}
Eigen::VectorXd Activation::sigmoidPrime(Eigen::VectorXd x) {
        x = x.array() * (1 - x.array());
        return x;
    }
void Activation::relu(Eigen::VectorXd& x) {
    x = x.cwiseMax(0.0);
}
Eigen::VectorXd Activation::reluPrime(Eigen::VectorXd& x) {
    x = (x.array() > 0).cast<double>();
    return x;
}
void Activation::softMax(Eigen::VectorXd& x) {
    x = x.array().exp() / x.array().exp().sum();
}
Eigen::MatrixXd Activation::softMaxPrime(Eigen::VectorXd x) {
    softMax(x);
    Eigen::MatrixXd derivative(x.size(), x.size());
    for(size_t i =0; i < x.size(); i++){
        for(size_t j = 0; j < x.size(); j++){
            if(i == j){
                derivative(i, j) = x(i) * (1 - x(i));
            }
            else{
                derivative(i, j) = x(i) * x(j);
            }
        }
    }
    return derivative;
}