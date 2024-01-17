# AutoEncoder
Implementation of an Autoencoder using c++ with the Eigen linear algebra library. 

## Requirements:

- C++ compiler 
- Eigen Library Downloaded (https://eigen.tuxfamily.org/index.php?title=Main_Page#Download)

## Running:

- Download the NeuralNetwork folder or clone this repository
- Create main file outside of NeuralNework folder where autoencoder will be used 
- Compile with " g++ NeuralNetwork/activation.cpp NeuralNetwork/layer.cpp NeuralNetwork/autoencoder.cpp "nameOfYourMainFile".cpp "

# Using autoencoder

- Ensure your main file includes the Eigen Dense folder location on your machine as well as the AutoEncoder.h File that is inside the NeuralNetwork folder
- Load in your data as a Eigen Matrix of doubles and your Labels as a standard vector of doubles
- Create autoencoder object by passing in the number of layers as an int followed by the size of input layer, size of hidden layers, and size of output layer all as integers
- Use train function by passing in the data to be trained as an Eigen Matrix "Eigen::MatrixXd", your labels for the data as a std vector of doubles,
    your learning rate as a double, the number of epochs as an int, and your batch sizes as an int
- View the mnist.cpp file for reference




