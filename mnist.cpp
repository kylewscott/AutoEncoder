#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <C:\eigen-3.4.0\Eigen\Dense>

#define MAX_SAMPLES 10 // number of samples to use for experimental training
#define DEBUG_OUTPUT

//Activation functions
void sigmoid(Eigen::VectorXd& x){
    x = 1.0/(1.0 + (-x.array()).exp());
}
Eigen::VectorXd sigmoidPrime(Eigen::VectorXd x){
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
Eigen::MatrixXd softMaxPrime(Eigen::VectorXd x){
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
    void updateWeights(Eigen::MatrixXd newWeight){
        weight = newWeight;
    }
    void updateBiases(Eigen::VectorXd newBias){
        bias = newBias;
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
    Eigen::VectorXd target;
    Eigen::VectorXd error;
    public:
    //Constructor to initiate layers
    AutoEncoder(){
        //initialize layers
        layer1 = Layer(128, 784);
        layer2 = Layer(64, 128);
        layer3 = Layer(10, 64);
    }
    //function to train data
    void train(Eigen::MatrixXd m, std::vector<double> labels, double learningRate, int epochs){
        for(size_t i = 0; i < epochs; i++){
            double tot_err = 0;
            for(size_t sample = 0; sample < MAX_SAMPLES; ++sample){
                //set target
                int val = labels[sample];
                target = Eigen::VectorXd::Zero(10); 
                target(val) = 1.0;
                //Feed forward 
                z1 = (layer1.getWeight() * m.col(sample)) + layer1.getBias();
                sigmoid(z1);
                z2 = (layer2.getWeight() * z1)  + layer2.getBias();
                sigmoid(z2);
                z3 = (layer3.getWeight() * z2) + layer3.getBias();
                softMax(z3);
                //find error
                Eigen::VectorXd err = target - z3;;
                err.array() /= 10;
                tot_err += err.array().abs().sum();
                //Back Propagate
                Eigen::VectorXd delta1 = softMaxPrime(z3) * (target - z3); //10x1
                Eigen::VectorXd delta2 = (layer3.getWeight().transpose() * delta1).array() * sigmoidPrime(z2).array(); //64x1
                Eigen::VectorXd delta3 = (layer2.getWeight().transpose() * delta2).array() * sigmoidPrime(z1).array(); //128x1
                //Update Weights and Biases
                layer3.updateWeights(layer3.getWeight().array() - (learningRate * (delta1 * z2.transpose())).array());
                layer3.updateBiases(layer3.getBias() - (learningRate * delta1));
                layer2.updateWeights(layer2.getWeight().array() - (learningRate * (delta2 * z1.transpose())).array());
                layer2.updateBiases(layer2.getBias() - (learningRate * delta2));
                layer1.updateWeights(layer1.getWeight().array() - (learningRate * (delta3 * m.col(sample).transpose())).array());
                layer1.updateBiases(layer1.getBias() - (learningRate * delta3));
            }
            //Calculate accuracy
            double accuracy = (1-(tot_err/MAX_SAMPLES)) * 100;
            //Display accuracy of each epoch
            #ifdef DEBUG_OUTPUT
                std::cout << "epoch: " << i+1 << ", " << "classification accuracy: " << accuracy << "\n";
            #endif
        }
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

    AutoEncoder a;
    a.train(matrix, labels, 0.001, 10);
    
    return 0;

}

