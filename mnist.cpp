#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>


using namespace std;    

int main() {
    //bring in mnist file and store in nums
    string filePath = "mnist_train.csv";
    ifstream nums;

    //open file
    nums.open(filePath);

    //Take in the first row of labels
    string str;
    getline(nums, str);

    //create vector to store values
    vector <int> column;

    //add first column of data into column
    while(getline(nums, str)){
        getline(nums, str, ',');
        //convert string to int
        stringstream s;
        s  << str;
        int number;
        s >> number;
        //put int into column
        column.push_back(number);
    }

    //create matrix for column to go into
    double matrix[28][28];
    int k = 0;

    //add values from column to matrix
    for(int i = 0; i < 28; i++){
        for(int j = 0; j < 28; j++){
            //normalize data
            double num = column[k] / 255.0;
            matrix[i][j] = num;
            k++;
        }
    }

    //print out matrix
    for(int i = 0; i < 28; i++){
        for(int j = 0; j < 28; j++){
            cout << matrix[i][j] << " ";
        }
    }

    return 0;

}