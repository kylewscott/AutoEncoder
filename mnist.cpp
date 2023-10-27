#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <C:\eigen-3.4.0\Eigen\Dense>
#include "AutoEncoder.h"

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

//Function to put specified digit into digit.csv 
void plotDigitInput(Eigen::MatrixXd m, int index){
    std::ofstream outputfile("digit.csv", std::ios::out);
    for(size_t i = 0; i < 784; i++){
        outputfile << m.coeff(i, index) << ",";
    }
    outputfile.close();
}

int main() {
    //load mnist data and labels
    std::vector<double> data = loadData("mnist_train.csv", "mnist");
    std::vector<double> labels = loadData("mnist_train_targets.csv", "labels");
    //Turn data into a matrix and print out
    Eigen::MatrixXd matrix = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data.data(), numRow, data.size() / numRow);
    //put specidifed digit into digit.csv
    plotDigitInput(matrix, 1);
    //Create autoencoder with a lr of 0.001 and 10 epochs
    AutoEncoder a(0.001, 10);
    //train the mnist data
    a.train(matrix, labels);
    
    return 0;
}

