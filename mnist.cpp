#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <C:\eigen-3.4.0\Eigen\Dense>


//Function to load in data as 1d vector
int numRow;
std::vector<double> loadData(std::string filePath, std::string info){
    //open file
    std::ifstream nums;
    nums.open(filePath);

    //Take in the first row 
    std::string str;
    getline(nums, str);

    //1d vector to store data
    std::vector<double> data;

    //Getting mnist
    if(info == "mnist"){
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
    }
    //Getting labels
    if(info == "labels"){
        while(nums >> str){
        int temp;
        std::stringstream s(str);
        while(s >> temp){
            std::string stemp;
            double number = temp;
            data.push_back(number);
            getline(s, stemp);
        }
    }
    } 
    return data;
}

//Activation functions
void sigmoid(Eigen::VectorXd& x){
    x = 1.0/(1.0 + (-x.array()).exp());
}
Eigen::VectorXd sigmoidPrime(Eigen::VectorXd& x){
    x = x.array() * (1 - x.array());
    return x;
}

void relu(Eigen::VectorXd& x){
    x = x.cwiseMax(0.0);
}
Eigen::VectorXd reluPrime(Eigen::VectorXd& x){
    x = (x.array() > 0).cast<double>();
    return x;
}

void softMax(Eigen::VectorXd& x){
    x = x.array().exp() / x.array().exp().sum();
}
Eigen::MatrixXd softMaxPrime(Eigen::VectorXd& x){
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
    sigmoid(z1);
    //input to hidden layer 2 size of 64
    Eigen::VectorXd z2 = (weight2 * z1) + bias2;
    sigmoid(z2);
    //input to output size of 10
    Eigen::VectorXd z3 = (weight3 * z2) + bias3;
    softMax(z3);

    //backward propagate

    Eigen::VectorXd error = target - z3;
    Eigen::VectorXd delta1 = softMaxPrime(z3) * error; //10x1
    Eigen::VectorXd delta2 = (weight3.transpose() * delta1).array() * sigmoidPrime(z2).array(); //64x1
    Eigen::VectorXd delta3 = (weight2.transpose() * delta2).array() * sigmoidPrime(z1).array(); //128x1

    //update weights and biases
    weight3 = (weight3.array() + learningRate).array() * (delta1 * z2.transpose()).array(); //10x64
    bias3 = (bias3.array() + learningRate).array() * delta1.array();    //10x1

    weight2 = (weight2.array() + learningRate).array() * (delta2 * z1.transpose()).array();  //64x128
    bias2 = (bias2.array() + learningRate).array() * delta2.array();   //64x1

    weight1 = (weight1.array() + learningRate).array() * (delta3 * m.transpose()).array();  //128x784
    bias1 = (bias1.array() + learningRate).array() * delta3.array();   //128x1

    //Calculate total error
    double totalError = std::sqrt(error.array().square().sum());
    double accuracy = (1 - totalError / m.cols()) * 100;
    std::cout << totalError << " " << accuracy;

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
    std::vector<double> data = loadData("mnist_train.csv", "mnist");
    std::vector<double> labels = loadData("mnist_train_targets.csv", "labels");
    //Creating target
    int value = labels[1];
    Eigen::VectorXd target(10, 1);
    target = Eigen::VectorXd::Zero(10,1);
    target(value,0) = 1.0;

    //Turn data into a matrix and print out
    Eigen::MatrixXd matrix = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data.data(), numRow, data.size() / numRow);

    //Call autoencoder function with mnist dataset
    autoencoder(matrix.col(1), target, 0.001, 10);
    //put specidifed digit into digit.csv
    plotDigitInput(matrix, 1);
    
    return 0;

}

// *** TO DO ***
//Replace all for loops with lambdas
//Create classes for the layers and autoencoder
//how to not use for loops for softMaxPrime
//figure out how to use derivative activation functions as voids
//clean up label loading
