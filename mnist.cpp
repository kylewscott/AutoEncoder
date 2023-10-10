// #include <iostream>
// #include <fstream>
// #include <string>
// #include <sstream>
// #include <vector>
// #include <C:\eigen-3.4.0\Eigen\Dense>

// //Class for layers
// class Layer {
//     public:
//         Eigen::MatrixXd weight;
//         Eigen::VectorXd bias;
//         Eigen::VectorXd layer;

//         Layer(int row, int col, Eigen::VectorXd input) {
//             //initialize weights and biases for layer
//             weight = Eigen::MatrixXd::Random(row, col);
//             weight = weight.array().abs();
//             bias = Eigen::VectorXd::Random(row);
//             layer = (weight * input) + bias;
//         }

//         //Forward pass function to be applied on each layer
//         Eigen::VectorXd forward(Eigen::MatrixXd weight, Eigen::VectorXd bias, Eigen::VectorXd input){
//             Eigen::VectorXd layer = (weight * input) + bias;
//             return layer;
//         }
// };

// //Class for activation functions and their derivatives
// class Activate {
//     public:
//         void sigmoid(Eigen::VectorXd& x){
//         x = 1.0/(1.0 + (-x.array()).exp());
//         }   
//         Eigen::VectorXd sigmoidPrime(Eigen::VectorXd& x){
//             x = x.array() * (1 - x.array());
//             return x;
//         }

//         void relu(Eigen::VectorXd& x){
//             x = x.cwiseMax(0.0);
//         }
//         Eigen::VectorXd reluPrime(Eigen::VectorXd& x){
//             x = (x.array() > 0).cast<double>();
//             return x;
//         }

//         void softMax(Eigen::VectorXd& x){
//             x = x.array().exp() / x.array().exp().sum();
//         }
//         Eigen::MatrixXd softMaxPrime(Eigen::VectorXd& x){       // **Needs work**
//             softMax(x);
//             Eigen::MatrixXd derivative(x.size(), x.size());
//             for(size_t i =0; i < x.size(); i++){
//                 for(size_t j = 0; j < x.size(); j++){
//                     if(i == j){
//                         derivative(i, j) = x(i) * (1 - x(i));
//                     }
//                     else{
//                         derivative(i, j) = x(i) * x(j);
//                     }
//                 }
//             }
//             return derivative;
//             // softMax(x);
//             // Eigen::MatrixXd derivative(x.size(), x.size());
//             // for(size_t i =0; i < x.size(); i++){
//             //     for(size_t j = 0; j < i; ++j)
//             //         derivative(i, j) = x(i) * x(j);
//             // }
//             // derivative.diagonal().array() = x.array() * (1 - x.array());
//             // derivative.upperTri() = derivative.lowerTri().transpose(); // DOES NOT WORK
//             // return derivative;
//         }
// };
// //Function to load in data as 1d vector
// int numRow;
// std::vector<double> loadData(std::string filePath, std::string info){
//     //open file
//     std::ifstream nums;
//     nums.open(filePath);

//     //Take in the first row 
//     std::string str;
//     getline(nums, str);

//     //1d vector to store data
//     std::vector<double> data;

//     //Getting mnist
//     if(info == "mnist"){
//         while(nums >> str){
//             int temp;
//             std::stringstream s(str);
//             while(s >> temp){
//                 std::string stemp;
//                 double number = temp / 255.0; //normalize data
//                 data.push_back(number);
//                 getline(s, stemp, ',');
//             }
//             numRow++;
//         }
//     }
//     //Getting labels
//     if(info == "labels"){
//         while(nums >> str){
//         int temp;
//         std::stringstream s(str);
//         while(s >> temp){
//             std::string stemp;
//             double number = temp;
//             data.push_back(number);
//             getline(s, stemp);
//         }
//     }
//     } 
//     return data;
// }

// //Autoencoder
// void autoencoder(Eigen::MatrixXd m, std::vector<double> labels, double learningRate, int epochs){
//     for(size_t i = 0; i < epochs; i++){
//         double totalError = 0.0;
//         for(size_t n = 0; n < 10; n++){
//             //Create target vector
//             int val = labels[n];
//             Eigen::VectorXd target = Eigen::VectorXd::Zero(10); 
//             target(val) = 1.0;

//             Activate a;
//             //initialize layers and forward pass
//             Layer layer1(128, 784, m.col(n));
//             a.sigmoid(layer1.layer);
//             Layer layer2(64, 128, layer1.layer);
//             a.sigmoid(layer2.layer);
//             Layer layer3(10, 64, layer2.layer);
//             a.softMax(layer3.layer);

//             //Find Error
//             Eigen::VectorXd error = layer3.layer - target;
//             totalError += std::sqrt(error.array().square().sum());
//             //std::cout << error << "\n\n";

//             //Back Propagate
//             Eigen::VectorXd delta1 = a.softMaxPrime(layer3.layer) * error; //10x1
//             Eigen::VectorXd delta2 = (layer3.weight.transpose() * delta1).array() * a.sigmoidPrime(layer2.layer).array(); //64x1
//             Eigen::VectorXd delta3 = (layer2.weight.transpose() * delta2).array() * a.sigmoidPrime(layer1.layer).array(); //128x1

//             Eigen::MatrixXd grad_weight3 = (delta1 * layer2.layer.transpose());
//             Eigen::MatrixXd grad_weight2 =  (delta2 * layer1.layer.transpose());
//             Eigen::MatrixXd grad_weight1 = (delta3 * m.col(n).transpose());

//             layer3.weight = layer3.weight - (learningRate * grad_weight3);
//             layer3.bias = layer3.bias - (learningRate * delta1);
//             layer2.weight = layer2.weight - (learningRate * grad_weight2);
//             layer2.bias = layer2.bias - (learningRate * delta2);
//             layer1.weight = layer1.weight - (learningRate * grad_weight1);
//             layer1.bias = layer1.bias - (learningRate * delta3);
            

//             //update weights and biases
//             //layer3.weight = (layer3.weight.array() + learningRate).array() * (delta1 * layer2.layer.transpose()).array(); //10x64            layer3.bias = (layer3.bias.array() + learningRate).array() * delta1.array();    //10x1
//             // layer3.weight = layer3.weight + (learningRate * (delta1 * layer2.layer.transpose()));
//             // layer3.bias = layer3.bias + (learningRate * delta1);

//             // // layer2.weight = (layer2.weight.array() + learningRate).array() * (delta2 * layer1.layer.transpose()).array();  //64x128
//             // // layer2.bias = (layer2.bias.array() + learningRate).array() * delta2.array();   //64x1
//             // layer2.weight = layer2.weight + (learningRate * (delta2 * layer1.layer.transpose()));
//             // layer2.bias = layer2.bias + (learningRate * delta2);

//             // // layer1.weight = (layer1.weight.array() + learningRate).array() * (delta3 * m.col(n).transpose()).array();  //128x784
//             // // layer1.bias = (layer1.bias.array() + learningRate).array() * delta3.array();   //128x1
//             // layer1.weight = layer1.weight + (learningRate * (delta3 * m.col(n).transpose()));
//             // layer1.bias = layer1.bias + (learningRate * delta3);
//         }
//         //Calculate accuracy of epoch
//         std::cout << totalError << '\n';
//         double accuracy = ((1 - totalError) / 10) * 100;
//         std::cout << "epoch: " << i+1 << ", " << "classification accuracy: " << accuracy << "\n";
//     }
// }

// //Function to put specified digit into digit.csv 
// void plotDigitInput(Eigen::MatrixXd m, int index){
//     std::ofstream outputfile("digit.csv", std::ios::out);
//     for(int i = 0; i < 784; i ++){
//         outputfile << m.coeff(i, index) << ",";
//     }
//     outputfile.close();
// }

// int main() {
//     //load mnist data and labels
//     std::vector<double> data = loadData("mnist_train.csv", "mnist");
//     std::vector<double> labels = loadData("mnist_train_targets.csv", "labels");
//     //Turn data into a matrix and print out
//     Eigen::MatrixXd matrix = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data.data(), numRow, data.size() / numRow);

//     //Call autoencoder function with mnist dataset
//     autoencoder(matrix, labels, 0.001, 10);
//     //put specidifed digit into digit.csv
//     plotDigitInput(matrix, 1);
    
//     return 0;

// }
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
        layer1 = Layer(128, 784);
        layer2 = Layer(64, 128);
        layer3 = Layer(10, 64);
    }
    void setWeights(){
        weight1 = layer1.getWeight();
        weight2 = layer2.getWeight();
        weight3 = layer3.getWeight();
    }
    void setBiases(){
        bias1 = layer1.getBias();
        bias2 = layer2.getBias();
        bias3 = layer3.getBias();
    }
    void setTarget(int val){
        target = Eigen::VectorXd::Zero(10); 
        target(val) = 1.0;
    }
    void feedForward(Eigen::VectorXd input){
        z1 = (weight1 * input) + bias1;
        sigmoid(z1);
        z2 = (weight2 * z1)  + bias2;
        sigmoid(z2);
        z3 = (weight3 * z2) + bias3;
        softMax(z3);
    }
    Eigen::VectorXd getError(){
        return target - z3;
    }
    void updateWeight(Eigen::MatrixXd &weight, double learningRate, Eigen::VectorXd input, Eigen::VectorXd delta){
        weight = weight.array() - (learningRate * (delta * input.transpose())).array();
    }
    void updateBias(Eigen::VectorXd &bias, double learningRate, Eigen::VectorXd delta){
        bias = bias - (learningRate * delta);
    }
    void backPropagate(double learningRate, Eigen::VectorXd input){
        Eigen::VectorXd delta1 = softMaxPrime(z3) * (target - z3); //10x1
        Eigen::VectorXd delta2 = (weight3.transpose() * delta1).array() * sigmoidPrime(z2).array(); //64x1
        Eigen::VectorXd delta3 = (weight2.transpose() * delta2).array() * sigmoidPrime(z1).array(); //128x1

        std::cout << weight3 << "\n\n";
        updateWeight(weight3, learningRate, z2, delta1);
        updateBias(bias3, learningRate, delta1);
        updateWeight(weight2, learningRate, z1, delta2);
        updateBias(bias2, learningRate, delta2);
        updateWeight(weight1, learningRate, input,delta3);
        // weight3 = weight3.array() - (learningRate * (delta1 * z2.transpose())).array();
        // bias3 = (bias3 - (learningRate * delta1));
        // weight2 = weight2.array() - (learningRate * (delta2 * z1.transpose())).array();
        // bias2 = (bias2 - (learningRate * delta2));
        // weight1 = weight1.array() - (learningRate * (delta3 * input.transpose())).array();
        // bias1 = (bias1 - (learningRate * delta3));
        std::cout << weight3 << "\n\n";
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

//Autoencoder
void autoencoder(Eigen::MatrixXd m, std::vector<double> labels, double learningRate, int epochs){
    for(size_t i = 0; i < epochs; i++){
        double totalError = 0.0;
        for(size_t n = 0; n < 10; n++){
            //create target
            int val = labels[n];
            Eigen::VectorXd target = Eigen::VectorXd::Zero(10); 
            target(val) = 1.0;

            //Create Layers and Forward pass
            // Layer layer1(128, 784);
            // Eigen::VectorXd z1 = layer1.getLayer();
            // sigmoid(z1);
            // Layer layer2(64, 128);
            // Eigen::VectorXd z2 = layer2.getLayer();
            // sigmoid(z2);
            // Layer layer3(10, 64);
            // Eigen::VectorXd z3 = layer3.getLayer();
            // softMax(z3);

            // Eigen::MatrixXd weight1 = layer1.getWeight();
            // Eigen::VectorXd bias1 = layer1.getBias();
            // Eigen::MatrixXd weight2 = layer2.getWeight();
            // Eigen::VectorXd bias2 = layer2.getBias();
            // Eigen::MatrixXd weight3 = layer3.getWeight();
            // Eigen::VectorXd bias3 = layer3.getBias();


            //Calculate error
            // Eigen::VectorXd error = (target - z3);
            // totalError += std::sqrt(error.array().square().sum());

            // //back propagate
            // Eigen::VectorXd delta1 = softMaxPrime(z3) * (target - z3); //10x1
            // Eigen::VectorXd delta2 = (weight3.transpose() * delta1).array() * sigmoidPrime(z2).array(); //64x1
            // Eigen::VectorXd delta3 = (weight2.transpose() * delta2).array() * sigmoidPrime(z1).array(); //128x1

            // //update weights and biases   **Weights not actually updating**
            // weight3 = weight3.array() - (learningRate * (delta1 * z2.transpose())).array();
            // bias3 = (bias3 - (learningRate * delta1));
            // weight2 = weight2.array() - (learningRate * (delta2 * z1.transpose())).array();
            // bias2 = (bias2 - (learningRate * delta2));
            // weight1 = weight1.array() - (learningRate * (delta3 * m.col(n).transpose())).array();
            // bias1 = (bias1 - (learningRate * delta3));
            // layer3.updateWeight(learningRate, delta1, z2);
            // layer3.updateBias(learningRate, delta1);

            // layer2.updateWeight(learningRate, delta2, z1);
            // layer2.updateBias(learningRate, delta2);

            // layer1.updateWeight(learningRate, delta3, m.col(n));
            // layer1.updateBias(learningRate, delta3);
        }
        //Calculate accuracy
        double accuracy = ((1 - totalError) / 10) * 100;
        std::cout << "epoch: " << i+1 << ", " << "classification accuracy: " << accuracy << "\n";
    }
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

    //Call autoencoder function with mnist dataset
    // autoencoder(matrix, labels, 9, 10);
    //put specidifed digit into digit.csv
    plotDigitInput(matrix, 1);

    int epochs = 10;
    AutoEncoder a;
    for(size_t i = 0; i < epochs; i++){
        double totalError = 0.0;
        for(size_t n = 0; n < 10; n++){
            a.setTarget(labels[n]);
            a.setWeights();
            a.setBiases();
            a.feedForward(matrix.col(n));

            Eigen::VectorXd error = a.getError();
            totalError += std::sqrt(error.array().square().sum());

            a.backPropagate(0.01, matrix.col(n));
        }
        double accuracy = ((1-totalError) / 10) * 100;
        std::cout << "epoch: " << i+1 << ", " << "classification accuracy: " << accuracy << "\n";
    }
    
    return 0;

}


// *** TO DO ***
//Create classes for the layers and autoencoder
//how to not use for loops for softMaxPrime
//figure out how to use derivative activation functions as voids
//Create stop condition
