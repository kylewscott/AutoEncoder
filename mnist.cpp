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

    //vector<vector<int>> matrix;
    int matrix[28][28];
    int k = 0;

    for(int i = 0; i < 28; i++){
        for(int j = 0; j < 28; j++){
            matrix[i][j] = column[k];
            k++;
        }
    }

    for(int i = 0; i < 28; i++){
        for(int j = 0; j < 28; j++){
            cout << matrix[i][j];
        }
    }




    //testing
    // //print out column
    // for(int i = 0; i < column.size(); i++){
    //     cout << column[i] << " ";
    // }
    // //double check size
    // cout << ", " << column.size();

    return 0;

}