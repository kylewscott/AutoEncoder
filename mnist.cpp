#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <autoEncoder.h>


using namespace std;    

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
    }
    
    return data;

}

int main() {
    //load mnist
    vector<double> data = loadData("mnist_train.csv");

    //check data was put in correctly
    for(int i = 0; i < data.size(); i++){
        cout << data[i] << " ";
    }

    return 0;

}