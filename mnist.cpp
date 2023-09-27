#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <random>
#include <C:\eigen-3.4.0\Eigen\Dense>

using namespace std;    
using namespace Eigen;

//Function to load in data as 1d vector
int numRow;
vector<double> loadData(string filePath){
    //open file
    ifstream nums;
    nums.open(filePath);

    //Take in the first row of labels
    string str;
    getline(nums, str);

    //1d vector to store entire mnist dataset
    vector<double> data;

    //add mnist data to data
    while(nums >> str){
        int temp;
        stringstream s(str);
        while(s >> temp){
            string stemp;
            double number = temp / 255.0; //normalize data
            data.push_back(number);
            getline(s, stemp, ',');
        }
        numRow++;
    }
    
    return data;

}

//function to load in labels as 1d vector
vector<double> loadLabels(string filepath){
    //open file
    ifstream label;
    label.open(filepath);

    //take in first row
    string str;
    getline(label, str);

    //create output vector
    vector<double> labels;

    //add each label into label
    while(label >> str){
        int temp;
        stringstream s(str);
        while(s >> temp){
            string stemp;
            double number = temp;
            labels.push_back(number);
            getline(s, stemp);
        }
    }
    return labels;
}

//Activation functions
MatrixXd sigmoid(MatrixXd x){
    for(int i = 0; i < x.size(); i++){
        x(i) = (1/(1+exp(-x(i))));
    }
    return x;
}

MatrixXd sigmoidPrime(MatrixXd x){
    for(int i = 0; i < x.size(); i++){
        x(i) = x(i) * (1-x(i));
    }
    return x;
}

MatrixXd relu(MatrixXd x){
    for(int i = 0; i < x.size(); i++){
        x(i) = max(0.0, x(i));
    }
    return x;
}

MatrixXd reluPrime(MatrixXd x){
    for(int i = 0; i < x.size(); i++){
        if(x(i) < 0){
            x(i) = 0;
        }
        else{
            x(i) = 1;
        }
    }
    return x;
}

MatrixXd softMax(MatrixXd x){
    double sum;
    for(int i = 0; i < x.size(); i++){
        sum += exp(x(i));
    }
    for(int i = 0; i < x.size(); i++){
        x(i) = exp(x(i)) / sum;
    }
    return x;
}

MatrixXd softMaxPrime(MatrixXd x){
    MatrixXd softmax = softMax(x);
    MatrixXd derivative;
    for(int i = 0; i < softmax.size(); i++){
        x(i) = softmax(i) * (1 - softmax(i));   
    }
    return x;
}

//Autoencoder
void autoencoder(MatrixXd m, MatrixXd target, double learningRate, int epochs){
    //Create Random number generator 0 - 1
    random_device rand;
    mt19937 gen(rand());
    uniform_real_distribution<double> dis(0.0, 1.0);
    //Create weights for each layer
    MatrixXd weight1(128, 784);
    for(int i = 0; i < 128; i++){
        for(int j = 0; j < 784; j++){
            weight1(i, j) = dis(gen);
        }
    }
    MatrixXd weight2(64, 128);
    for(int i = 0; i < 64; i++){
        for(int j = 0; j < 128; j++){
            weight2(i, j) = dis(gen);
        }
    }
    MatrixXd weight3(10, 64);
    for(int i = 0; i < 10; i++){
        for(int j = 0; j < 64; j++){
            weight3(i, j) = dis(gen);
        }
    }
    //Create biases for each layer
    MatrixXd bias1(128,1);
    for(int i = 0; i < 128; i++){
        bias1(i, 0) = dis(gen);
    }
    MatrixXd bias2(64,1);
    for(int i = 0; i < 64; i++){
        bias2(i, 0) = dis(gen);
    }
    MatrixXd bias3(10,1);
    for(int i = 0; i < 10; i++){
        bias3(i, 0) = dis(gen);
    }

    //Forward propagate
    //input layer size of 784
    MatrixXd a0 = m;
    //input to hidden layer 1 size of 128
    MatrixXd z1 = (weight1 * a0) + bias1;
    MatrixXd a1 = sigmoid(z1);
    //input to hidden layer 2 size of 64
    MatrixXd z2 = (weight2 * a1) + bias2;
    MatrixXd a2 = sigmoid(z2);
    //input to output size of 10
    MatrixXd z3 = (weight3 * a2) + bias3;
    MatrixXd a3 = softMax(z3);

    //backward propagate

    MatrixXd error = a3 - target;
    MatrixXd delta1(10,1);
    for(int i = 0; i < 10; i++){
        delta1(i) = error(i) * softMaxPrime(z3)(i);
    }
    MatrixXd deltaW3 = delta1 * a2.transpose();
    MatrixXd deltaB3 = delta1;

    MatrixXd temp = weight3.transpose() * delta1;
    MatrixXd delta2(64,1);
    for(int i = 0; i < 64; i++){
        delta2(i) = temp(i) * sigmoidPrime(z2)(i);
    }
    MatrixXd deltaW2 = delta2 * a1.transpose();
    MatrixXd deltaB2 = delta2;
    
    temp = weight2.transpose() * delta2;
    MatrixXd delta3(128,1);
    for(int i = 0; i < 128; i++){
        delta3(i) = temp(i) * sigmoidPrime(z1)(i);
    }
    MatrixXd deltaW1 = delta3 * a0.transpose();
    MatrixXd deltaB1 = delta3;

    //update weights and biases
    

}

//Function to put specified digit into digit.csv 
void plotDigitInput(MatrixXd m, int index){
    ofstream outputfile("digit.csv", ios::out);
    for(int i = 0; i < 784; i ++){
        outputfile << m.coeff(i, index) << ",";
    }
    outputfile.close();
}

int main() {
    //load mnist data and labels
    vector<double> data = loadData("mnist_train.csv");
    vector<double> labels = loadLabels("mnist_train_targets.csv");
    int value = labels[25];
    MatrixXd target(10, 1);
    target = MatrixXd::Zero(10,1);
    target(value,0) = 1.0;
    //cout << target;

    //Turn data into a matrix and print out
    MatrixXd matrix = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(data.data(), numRow, data.size() / numRow);

    //Call autoencoder function with mnist dataset
    autoencoder(matrix.col(1), target, 0.01, 10);
    //put specidifed digit into digit.csv
    //cout << labels[25] << "\n\n";
    plotDigitInput(matrix, 25);
    

    return 0;

}