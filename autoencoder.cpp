#include "AutoEncoder.h"
#define MAX_SAMPLES 10
#define DEBUG_OUTPUT
//Constructor
AutoEncoder::AutoEncoder(double lr, int epoch) {
    layers = Layer(4, 784, 128, 64, 10);
    learningRate = lr;
    epochs = epoch;
}
void AutoEncoder::train(Eigen::MatrixXd m, std::vector<double> labels) {
    Activation a;
    for(size_t i = 0; i < epochs; i++){
        layers.resetTotErr();
        for(size_t sample = 0; sample < MAX_SAMPLES; ++sample){
            //set target
            int val = labels[sample];
            target = Eigen::VectorXd::Zero(10); 
            target(val) = 1.0;
            //Feed forward and backward
            layers.feedForward(m.col(sample), target);
            layers.backwardsPropagation(m.col(sample), learningRate);
        }
        //Calculate accuracy
        double accuracy = (1-(layers.getTotErr()/MAX_SAMPLES)) * 100;
        //Display accuracy of each epoch
        #ifdef DEBUG_OUTPUT
            std::cout << "epoch: " << i+1 << ", " << "classification accuracy: " << accuracy << "\n";
        #endif
        }
}