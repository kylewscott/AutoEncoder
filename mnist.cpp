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
        stringstream s;
        s  << str;
        int number;
        s >> number;
        column.push_back(number);
    }
    //print out column
    for(int i = 0; i < column.size(); i++){
        cout << column[i] << " ";
    }
    //double check size
    cout << ", " << column.size();

    return 0;

}