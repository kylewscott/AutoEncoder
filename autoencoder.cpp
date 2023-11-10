#include "AutoEncoder.h"
//number of samples from dataset
#define MAX_SAMPLES 32000
//toggle accuracy output   
#define DEBUG_OUTPUT 
//Construct autoenocder based on the given layers
AutoEncoder::AutoEncoder(int numLayers, int sizes...) {
    layers = Layer(numLayers, sizes);
}
//pass in data, labels, learningRate, epochs, batchSize for training
void AutoEncoder::train(Eigen::MatrixXd m, std::vector<double> labels, double lr, int epochs, int batchSize) {
    Activation a;
    //loop through epochs
    for(size_t i = 0; i < epochs; i++){
        Layer localLayers = layers;
        layers.resetTotErr();
        //loop through batches
        for(size_t batch = 0; batch < MAX_SAMPLES; batch += batchSize){
            //loop through samples in each batch
            for(size_t sample = 0; sample < batchSize; ++sample){
                //set target
                int val = labels[sample+batch];
                target = Eigen::VectorXd::Zero(10); 
                target(val) = 1.0;
                //Feed forward and backward and update
                layers.feedForward(m.col(sample+batch), target);
                layers.backwardsPropagation(m.col(sample+batch), learningRate);
                layers.updateWeightsBiases(learningRate);
            }
        }
        //Calculate accuracy
        double accuracy = (1-(layers.getTotErr()/MAX_SAMPLES)) * 100;
        //Display accuracy of each epoch
        #ifdef DEBUG_OUTPUT
            std::cout << "epoch: " << i+1 << ", " << "classification accuracy: " << accuracy << "\n";
        #endif
    }
}

//****** Things to work on **********
//  -Parallel processing
//  -Possibly allowing user to enter which activation function to use