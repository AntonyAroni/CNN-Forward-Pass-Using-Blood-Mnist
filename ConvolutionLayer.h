//
// Created by antony on 6/15/25.
//

#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H

#include "ConvolutionKernel.h"
#include "ActivationFunction.h"
#include "Enums.h"
#include <vector>

class ConvolutionLayer {
private:
    ConvolutionKernel kernel;
    ActivationType activation_type;
    int stride;
    int padding;

public:
    ConvolutionLayer(int in_channels, int out_channels, int kernel_size,
                     ActivationType act_type = ActivationType::RELU, int s = 1, int p = 0);

    std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<std::vector<float>>>& input);

    void printKernelInfo() const;
};

#endif // CONVOLUTION_LAYER_H