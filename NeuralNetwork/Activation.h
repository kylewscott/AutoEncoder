#include <iostream>
#include <C:\eigen-3.4.0\Eigen\Dense>
class Activation {
    public:
    void sigmoid(Eigen::VectorXd& x);
    Eigen::VectorXd sigmoidPrime(Eigen::VectorXd x);
    void relu(Eigen::VectorXd& x);
    Eigen::VectorXd reluPrime(Eigen::VectorXd& x);
    void softMax(Eigen::VectorXd& x);
    Eigen::MatrixXd softMaxPrime(Eigen::VectorXd x);
};