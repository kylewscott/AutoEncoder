#include "AutoEncoder.h"
//number of samples from dataset
#define MAX_SAMPLES 3200
//toggle accuracy output   
#define DEBUG_OUTPUT 
//Construct autoenocder based on the given layers
AutoEncoder::AutoEncoder(int numLayers, int sizes...) {
    //load sizes into vector to feed to layer object
    std::va_list args;
    va_start(args, sizes);
    layerSizes.push_back(sizes);
    for(size_t i = 0; i < numLayers-1; ++i){
        layerSizes.push_back(va_arg(args, int));
    }
    va_end(args);
    //create layers 
    layers = Layer(numLayers, layerSizes);
}
//pass in data, labels, learningRate, epochs, batchSize for training
void AutoEncoder::train(Eigen::MatrixXd m, std::vector<double> labels, double lr, int epochs, int batchSize) {
    Activation a;
    //loop through epochs
    for(size_t i = 0; i < epochs; i++){
        layers.resetTotErr();
        //loop through batches
        for(size_t batch = 0; batch < MAX_SAMPLES; batch += batchSize){
            //loop through samples in each batch
            //#pragma omp parallel for
            for(size_t sample = 0; sample < batchSize; ++sample){
                //set target
                int val = labels[sample+batch];
                target = Eigen::VectorXd::Zero(10); 
                target(val) = 1.0;
                //Feed forward and backward and update
                layers.feedForward(m.col(sample+batch), target);
                layers.backwardsPropagation(m.col(sample+batch), lr);
                layers.updateWeightsBiases(lr);
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