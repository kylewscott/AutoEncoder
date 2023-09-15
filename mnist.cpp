#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
// #include <autoEncoder.h>
#include <C:\eigen-3.4.0\Eigen\Dense>



using namespace std;    
using namespace Eigen;

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

int main() {
    //load mnist
    vector<double> data = loadData("mnist_train.csv");

    //Turn data into a matrix and print out
    MatrixXd matrix = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(data.data(), numRow, data.size() / numRow);
    cout << matrix;

    return 0;

}