#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
//#include <autoEncoder.h>
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

//Activation functions
MatrixXd sigmoid(MatrixXd x){
    for(int i = 0; i < x.size(); i++){
        x(i) = (1/(1+exp(-x(i))));
    }
    return x;
}

double sigmoidPrime(double x){
    return x * (1-x);
}

MatrixXd relu(MatrixXd x){
    for(int i = 0; i < x.size(); i++){
        x(i) = max(0.0, x(i));
    }
    return x;
}

double reluPrime(double x){
    if(x < 0){
        return 0;
    }
    else{
        return 1;
    }
}

//Autoencoder
void autoencoder(MatrixXd m, double learningRate, int epochs){
    //Weight 1, weights in between input and hidden layer 1
    MatrixXd weight1 = MatrixXd::Random(128, 784);
    weight1 = (weight1 +  MatrixXd::Constant(128, 784, 1.))*sqrt(2.0/784);
    //Weight 2, weights in between hidden layer 1 and 2
    MatrixXd weight2 = MatrixXd::Random(64, 128);
    weight2 = (weight2 + MatrixXd::Constant(64, 128, 1.))*sqrt(2.0/128);
    //Weight 3, weights in between hidden layer 2 and output
    MatrixXd weight3 = MatrixXd::Random(10, 64);
    weight3 = (weight3 + MatrixXd::Constant(10, 64, 1.))*sqrt(2.0/64);

    //Forward propagate
    //input layer size of 784
    MatrixXd a0 = m;
    //input to hidden layer 1 size of 128
    MatrixXd z1 = weight1 * a0;
    MatrixXd a1 = relu(z1);
    //input to hidden layer 2 size of 64
    MatrixXd z2 = weight2 * a1;
    MatrixXd a2 = relu(z2);
    //input to output size of 10
    MatrixXd z3 = weight3 * a2;
    MatrixXd a3 = sigmoid(z3);

    cout << a3;
    

}

int main() {
    //load mnist
    vector<double> data = loadData("mnist_train.csv");

    //Turn data into a matrix and print out
    MatrixXd matrix = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(data.data(), numRow, data.size() / numRow);
    // for(int i = 0; i < 784; i++){
    //     cout << matrix.coeff(i,3) << ", ";
    // } 

    autoencoder(matrix.col(1), 0.01, 10);

    return 0;

}