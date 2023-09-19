#include <iostream>
#include <algorithm>
#include <math.h>
#include <vector>
#include <autoEncoder.h>

using namespace std;

//create activation functions and derivatives
double sigmoid(double x){
    return (1/(1+exp(-x)));
}

double sigmoidPrime(double x){
    return x * (1-x);
}

double relu(double x){
    return max(0.0, x);
}

double reluPrime(double x){
    if(x < 0){
        return 0;
    }
    else{
        return 1;
    }
}

int autoencoder(vector<double> mnist, double learningRate, int epochs){
    //initialize wieghts (He) and biases (0.01)
    //using 784 as that is the size of each layer 
    double bias = 0.01;
    double weight1 = ((rand() % 784-1) + 784) * (sqrt(2/(784-1)));
    double weight2 = ((rand() % 784-1) + 784) * (sqrt(2/(784-1)));
    double weight3 = ((rand() % 784-1) + 784) * (sqrt(2/(784-1)));

    for(int i = 0; i < epochs; i++){
        for(int n = 0; n < 784; n++){
            //forward
            double z1 = (mnist[i] * weight1) + bias;
            double a1 = relu(z1);
            double z2 = (a1* weight2) + bias;
            double a2 = relu(z2);
            double z3 = (a2 * weight3) + bias;
            double a3 = sigmoid(z3);
        }
        //then back propagate with error calculation
        //update the weights and biases
        //print out accuracy
    }

    return 0;
}


