//
// Created by antony on 6/15/25.
//

#include "ConvolutionKernel.h"
#include <random>

ConvolutionKernel::ConvolutionKernel(int in_ch, int out_ch, int k_size)
    : in_channels(in_ch), out_channels(out_ch), kernel_size(k_size) {
    initializeWeights();
}

void ConvolutionKernel::initializeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);

    weights.resize(out_channels);
    for (int i = 0; i < out_channels; i++) {
        weights[i].resize(in_channels);
        for (int j = 0; j < in_channels; j++) {
            weights[i][j].resize(kernel_size);
            for (int k = 0; k < kernel_size; k++) {
                weights[i][j][k].resize(kernel_size);
                for (int l = 0; l < kernel_size; l++) {
                    weights[i][j][k][l] = dist(gen);
                }
            }
        }
    }

    bias.resize(out_channels);
    for (int i = 0; i < out_channels; i++) {
        bias[i] = dist(gen);
    }
}

const std::vector<std::vector<std::vector<std::vector<float>>>>& ConvolutionKernel::getWeights() const {
    return weights;
}

const std::vector<float>& ConvolutionKernel::getBias() const {
    return bias;
}

int ConvolutionKernel::getKernelSize() const {
    return kernel_size;
}

int ConvolutionKernel::getInChannels() const {
    return in_channels;
}

int ConvolutionKernel::getOutChannels() const {
    return out_channels;
}