#include "AutoEncoder.h"
#define MAX_SAMPLES 10
#define DEBUG_OUTPUT
//Constructor
AutoEncoder::AutoEncoder() {
    // layer1 = Layer(256, 784);
    // layer2 = Layer(128, 256);
    // layer3 = Layer(32, 128);
    layer1 = Layer(128, 784);
    layer2 = Layer(64, 128);
    layer3 = Layer(10, 64);

}
void AutoEncoder::train(Eigen::MatrixXd m, std::vector<double> labels, double learningRate, int epochs) {
    Activation a;
    for(size_t i = 0; i < epochs; i++){
        double tot_err = 0;
        for(size_t sample = 0; sample < MAX_SAMPLES; ++sample){
            //set target
            int val = labels[sample];
            target = Eigen::VectorXd::Zero(10); 
            target(val) = 1.0;
            //Feed forward 
            layer1.feedForward(m.col(sample), "sigmoid");
            layer2.feedForward(z1, "sigmoid");
            layer3.feedForward(z2, "softMax");
            //find error
            Eigen::VectorXd err = z3-target;
            err.array() /= MAX_SAMPLES;
            tot_err += err.array().abs().sum();
            //Back Propagate
            Eigen::VectorXd delta1 = a.softMaxPrime(z3) * (z3-target); //10x1
            Eigen::VectorXd delta2 = (layer3.getWeight().transpose() * delta1).array() * a.sigmoidPrime(z2).array(); //64x1
            Eigen::VectorXd delta3 = (layer2.getWeight().transpose() * delta2).array() * a.sigmoidPrime(z1).array(); //128x1
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