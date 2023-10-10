#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <C:\eigen-3.4.0\Eigen\Dense>

//Activation functions
void sigmoid(Eigen::VectorXd& x){
    x = 1.0/(1.0 + (-x.array()).exp());
}
Eigen::VectorXd sigmoidPrime(Eigen::VectorXd& x){
    x = x.array() * (1 - x.array());
    return x;
}

void relu(Eigen::VectorXd& x){
    x = x.cwiseMax(0.0);
}
Eigen::VectorXd reluPrime(Eigen::VectorXd& x){
    x = (x.array() > 0).cast<double>();
    return x;
}

void softMax(Eigen::VectorXd& x){
    x = x.array().exp() / x.array().exp().sum();
}
Eigen::MatrixXd softMaxPrime(Eigen::VectorXd& x){
    softMax(x);
    Eigen::MatrixXd derivative(x.size(), x.size());
    for(size_t i =0; i < x.size(); i++){
        for(size_t j = 0; j < x.size(); j++){
            if(i == j){
                derivative(i, j) = x(i) * (1 - x(i));
            }
            else{
                derivative(i, j) = x(i) * x(j);
            }
        }
    }
    
    return derivative;
    // softMax(x);
    // Eigen::MatrixXd derivative(x.size(), x.size());
    // for(size_t i =0; i < x.size(); i++){
    //     for(size_t j = 0; j < i; ++j)
    //         derivative(i, j) = x(i) * x(j);
    // }
    // derivative.diagonal().array() = x.array() * (1 - x.array());
        // derivative.upperTri() = derivative.lowerTri().transpose(); // DOES NOT WORK
        // return derivative;
}

// //Class for layers
class Layer {
    private:
    Eigen::MatrixXd weight;
    Eigen::VectorXd bias;
    public:
    Layer(){}
    Layer(int row, int col) {
        //initialize weights and biases for layer
        weight = Eigen::MatrixXd::Random(row, col);
        weight = weight.array().abs();
        bias = Eigen::VectorXd::Random(row);
    }
    Eigen::MatrixXd getWeight(){
        return weight;
    }
    Eigen::VectorXd getBias(){
        return bias;
    }
};

class AutoEncoder {
    private:
    Layer layer1;
    Layer layer2;
    Layer layer3;
    Eigen::VectorXd z1;
    Eigen::VectorXd z2;
    Eigen::VectorXd z3;
    Eigen::MatrixXd weight1;
    Eigen::MatrixXd weight2;
    Eigen::MatrixXd weight3;
    Eigen::VectorXd bias1;
    Eigen::VectorXd bias2;
    Eigen::VectorXd bias3;
    Eigen::VectorXd target;
    Eigen::VectorXd error;
    public:
    AutoEncoder(){
        //initialize layers
        layer1 = Layer(128, 784);
        layer2 = Layer(64, 128);
        layer3 = Layer(10, 64);
    }
    void setWeights(){
        //set weights
        weight1 = layer1.getWeight();
        weight2 = layer2.getWeight();
        weight3 = layer3.getWeight();
    }
    void setBiases(){
        //set biases
        bias1 = layer1.getBias();
        bias2 = layer2.getBias();
        bias3 = layer3.getBias();
    }
    void setTarget(int val){
        //Set the target
        target = Eigen::VectorXd::Zero(10); 
        target(val) = 1.0;
    }
    void feedForward(Eigen::VectorXd input){
        //Forward propagate and activate
        z1 = (weight1 * input) + bias1;
        sigmoid(z1);
        z2 = (weight2 * z1)  + bias2;
        sigmoid(z2);
        z3 = (weight3 * z2) + bias3;
        softMax(z3);
    }
    Eigen::VectorXd getError(){
        //Calculate the error
        return target - z3;
    }
    void backPropagate(double learningRate, Eigen::VectorXd input){
        //Back propagatge
        Eigen::VectorXd delta1 = softMaxPrime(z3) * (target - z3); //10x1
        Eigen::VectorXd delta2 = (weight3.transpose() * delta1).array() * sigmoidPrime(z2).array(); //64x1
        Eigen::VectorXd delta3 = (weight2.transpose() * delta2).array() * sigmoidPrime(z1).array(); //128x1
        //Update Weights and Biases
        weight3 = weight3.array() - (learningRate * (delta1 * z2.transpose())).array();
        bias3 = (bias3 - (learningRate * delta1));
        weight2 = weight2.array() - (learningRate * (delta2 * z1.transpose())).array();
        bias2 = (bias2 - (learningRate * delta2));
        weight1 = weight1.array() - (learningRate * (delta3 * input.transpose())).array();
        bias1 = (bias1 - (learningRate * delta3));
    }
};

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
    for(int i = 0; i < 784; i ++){
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

    //Call autoencoder
    int epochs = 10;
    AutoEncoder a;
    //loop thorugh numbers for each epoch
    for(size_t i = 0; i < epochs; i++){
        double totalError = 0.0;
        for(size_t n = 0; n < 10; n++){
            //Set target, weights, and biases
            a.setTarget(labels[n]);
            a.setWeights();
            a.setBiases();  
            //forward propagate
            a.feedForward(matrix.col(n));
            //find error and add to totalError
            Eigen::VectorXd error = a.getError();
            totalError += std::sqrt(error.array().square().sum());
            //Back propagate
            a.backPropagate(0.01, matrix.col(n));
        }
        //Calculate accuracy of each epoch
        double accuracy = ((1-totalError) / 10) * 100;
        std::cout << "epoch: " << i+1 << ", " << "classification accuracy: " << accuracy << "\n";
    }
    
    return 0;

}

