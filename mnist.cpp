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
vector<int> loadLabels(string filepath){
    //open file
    ifstream label;
    label.open(filepath);

    //take in first row
    string str;
    getline(label, str);

    //create output vector
    vector<int> labels;

    //add each label into label
    while(label >> str){
        int temp;
        stringstream s(str);
        while(s >> temp){
            string stemp;
            int number = temp;
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

//Autoencoder
void autoencoder(MatrixXd m, vector<int> labels, double learningRate, int epochs){
    // //Weight 1, weights in between input and hidden layer 1
    // MatrixXd weight1 = MatrixXd::Random(128, 784);
    // weight1 = (weight1 +  MatrixXd::Constant(128, 784, 1.))*sqrt(2.0/784);
    // MatrixXd bias1 = MatrixXd::Random(128, 1);
    // bias1 = (bias1 + MatrixXd::Constant(128, 1, 1.));
    // //Weight 2, weights in between hidden layer 1 and 2
    // MatrixXd weight2 = MatrixXd::Random(64, 128);
    // weight2 = (weight2 + MatrixXd::Constant(64, 128, 1.))*sqrt(2.0/128);
    // MatrixXd bias2 = MatrixXd::Random(64, 1);
    // bias2 = (bias2 + MatrixXd::Constant(64, 1, 1.));
    // //Weight 3, weights in between hidden layer 2 and output
    // MatrixXd weight3 = MatrixXd::Random(10, 64);
    // weight3 = (weight3 + MatrixXd::Constant(10, 64, 1.))*sqrt(2.0/64);
    // MatrixXd bias3 = MatrixXd::Random(10, 1);
    // bias3 = (bias3 + MatrixXd::Constant(10, 1, 1.));
    // cout << weight1;

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
    cout << bias1 << "\n\n" << bias2 << "\n\n" << bias3;

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
    // cout << z3 << "\n\n" << a3;

    //compute losee

    //backward propagate

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
    vector<int> labels = loadLabels("mnist_train_targets.csv");

    //Turn data into a matrix and print out
    MatrixXd matrix = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(data.data(), numRow, data.size() / numRow);

    //Call autoencoder function with mnist dataset
    autoencoder(matrix.col(1), labels, 0.01, 10);
    //put specidifed digit into digit.csv
    cout << labels[25];
    plotDigitInput(matrix, 25);
    

    return 0;

}