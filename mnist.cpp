#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <random>
#include <C:\eigen-3.4.0\Eigen\Dense>

using namespace Eigen;

//Function to load in data as 1d vector
int numRow;
std::vector<double> loadData(std::string filePath){
    //open file
    std::ifstream nums;
    nums.open(filePath);

    //Take in the first row of labels
    std::string str;
    getline(nums, str);

    //1d vector to store entire mnist dataset
    std::vector<double> data;

    //add mnist data to data
    while(nums >> str){
        int temp;
        std::stringstream s(str);
        while(s >> temp){
            std::string stemp;
            double number = temp / 255.0; //normalize data
            data.push_back(number);
            getline(s, stemp, ',');
        }
        numRow++;
    }
    
    return data;

}

//function to load in labels as 1d vector
std::vector<double> loadLabels(std::string filepath){
    //open file
    std::ifstream label;
    label.open(filepath);

    //take in first row
    std::string str;
    getline(label, str);

    //create output vector
    std::vector<double> labels;

    //add each label into label
    while(label >> str){
        int temp;
        std::stringstream s(str);
        while(s >> temp){
            std::string stemp;
            double number = temp;
            labels.push_back(number);
            getline(s, stemp);
        }
    }
    return labels;
}

//Possible change to void functions?
//Activation functions
Eigen::VectorXd sigmoid(Eigen::VectorXd x){
    // try to do this in-place using an Array view of the matrix
    //x.array().exp().addTo(1).inverse();

    for(size_t i = 0; i < x.size(); i++){
        x(i) = (1/(1+exp(-x(i))));
    }
    return x;
}

Eigen::VectorXd sigmoidPrime(Eigen::VectorXd x){
    for(size_t i = 0; i < x.size(); i++){
        x(i) = x(i) * (1-x(i));
    }
    return x;
}

Eigen::VectorXd relu(Eigen::VectorXd x){
    for(size_t i = 0; i < x.size(); i++){
        x(i) = std::max(0.0, x(i));
    }
    return x;
}

Eigen::VectorXd reluPrime(Eigen::VectorXd x){
    for(size_t i = 0; i < x.size(); i++){
        if(x(i) < 0){
            x(i) = 0;
        }
        else{
            x(i) = 1;
        }
    }
    return x;
}

Eigen::VectorXd softMax(Eigen::VectorXd x){
    double sum;
    for(size_t i = 0; i < x.size(); i++){
        sum += exp(x(i));
    }
    for(size_t i = 0; i < x.size(); i++){
        x(i) = exp(x(i)) / sum;
    }
    return x;
}

Eigen::MatrixXd softMaxPrime(Eigen::VectorXd x){
    Eigen::VectorXd softmax = softMax(x);
    Eigen::MatrixXd derivative(10,10);
    for(size_t i =0; i < softmax.size(); i++){
        for(size_t j = 0; j < softmax.size(); j++){
            if(i == j){
                derivative(i, j) = softmax(i) * (1 - softmax(i));
            }
            else{
                derivative(i, j) = softmax(i) * softmax(j);
            }
        }
    }
    return derivative;
}

//Autoencoder
void autoencoder(Eigen::MatrixXd m, Eigen::VectorXd target, double learningRate, int epochs){
    //Create Random weights and biases 0-1
    Eigen::MatrixXd weight1 = Eigen::MatrixXd::Random(128, 784);
    weight1 = weight1.array().abs();

    Eigen::MatrixXd weight2 = Eigen::MatrixXd::Random(64, 128);
    weight2 = weight2.array().abs();

    Eigen::MatrixXd weight3 = Eigen::MatrixXd::Random(10, 64);
    weight3 = weight3.array().abs();

    Eigen::VectorXd bias1 = Eigen::VectorXd::Random(128);
    bias1 = bias1.array().abs();

    Eigen::VectorXd bias2 = Eigen::VectorXd::Random(64);
    bias2 = bias2.array().abs();

    Eigen::VectorXd bias3 = Eigen::VectorXd::Random(10);
    bias3 = bias3.array().abs();

    //Forward propagate
    //input to hidden layer 1 size of 128
    Eigen::VectorXd z1 = (weight1 * m) + bias1;
    Eigen::VectorXd a1 = sigmoid(z1);
    //input to hidden layer 2 size of 64
    Eigen::VectorXd z2 = (weight2 * a1) + bias2;
    Eigen::VectorXd a2 = sigmoid(z2);
    //input to output size of 10
    Eigen::VectorXd z3 = (weight3 * a2) + bias3;
    Eigen::VectorXd a3 = softMax(z3);

    //backward propagate
    Eigen::VectorXd error = a3 - target;
    Eigen::VectorXd delta1 = softMaxPrime(a3) * error; //10x1
    Eigen::VectorXd delta2 = (weight3.transpose() * delta1).array() * sigmoidPrime(a2).array(); //64x1
    Eigen::VectorXd delta3 = (weight2.transpose() * delta2).array() * sigmoidPrime(z1).array(); //128x1

    //update weights and biases
    weight3 = (weight3.array() + learningRate).array() * (delta1 * a2.transpose()).array(); //10x64
    bias3 = (bias3.array() + learningRate).array() * delta1.array();    //10x1

    weight2 = (weight2.array() + learningRate).array() * (delta2 * a1.transpose()).array();  //64x128
    bias2 = (bias2.array() + learningRate).array() * delta2.array();   //64x1

    weight1 = (weight1.array() + learningRate).array() * (delta3 * m.transpose()).array();  //128x784
    bias1 = (bias1.array() + learningRate).array() * delta3.array();   //128x1

    //Calculate total error
    double totalError = std::sqrt(error.array().square().sum());

}

//Function to put specified digit into digit.csv 
void plotDigitInput(Eigen::MatrixXd m, int index){
    std::ofstream outputfile("digit.csv", std::ios::out);
    for(int i = 0; i < 784; i ++){
        outputfile << m.coeff(i, index) << ",";
    }
    outputfile.close();
}

int main() {
    //load mnist data and labels
    std::vector<double> data = loadData("mnist_train.csv");
    std::vector<double> labels = loadLabels("mnist_train_targets.csv");
    int value = labels[25];
    Eigen::VectorXd target(10, 1);
    target = Eigen::VectorXd::Zero(10,1);
    target(value,0) = 1.0;

    //Turn data into a matrix and print out
    Eigen::MatrixXd matrix = Eigen::Map<Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>>(data.data(), numRow, data.size() / numRow);

    //Call autoencoder function with mnist dataset
    autoencoder(matrix.col(1), target, 0.01, 10);
    //put specidifed digit into digit.csv
    plotDigitInput(matrix, 25);
    
    return 0;

}

// *** TO DO ***
//Go through and take out unecessary lines
//Look into lambdas if needed
//Create classes for the layers and autoencoder
