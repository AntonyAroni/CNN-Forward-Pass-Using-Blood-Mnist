#ifndef CONVOLUTIONAL_NEURAL_NETWORK_H
#define CONVOLUTIONAL_NEURAL_NETWORK_H

#include "ConvolutionLayer.h"
#include "PoolingLayer.h"
#include "Enums.h"
#include <vector>

class ConvolutionalNeuralNetwork {
private:
    std::vector<ConvolutionLayer> conv_layers;
    std::vector<PoolingLayer> pool_layers;

public:
    void addConvolutionLayer(int in_channels, int out_channels, int kernel_size,
                           ActivationType activation = ActivationType::RELU);

    void addPoolingLayer(int pool_size, PoolingType type = PoolingType::MAX);

    std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<std::vector<float>>>& input);

    void printArchitecture() const;
};

#endif // CONVOLUTIONAL_NEURAL_NETWORK_H