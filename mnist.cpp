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

    //make sure file is opening
    if(nums.fail()){
        cout << "Unable to open file " << filePath;
        return 1;
    }

    //Take in the first row of labels
    string str;
    getline(nums, str);

    vector <string> column;

    while(getline(nums, str)){
        getline(nums, str, ',');
        column.push_back(str);
    }
    for(int i = 0; i < column.size(); i++){
        cout << column[i];
    }

    cout << ", " << column.size();

    return 0;

}