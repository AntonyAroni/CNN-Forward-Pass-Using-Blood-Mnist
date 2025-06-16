//
// Created by antony on 6/15/25.
//

#ifndef CONVOLUTION_KERNEL_H
#define CONVOLUTION_KERNEL_H

#include <vector>

class ConvolutionKernel {
private:
    std::vector<std::vector<std::vector<std::vector<float>>>> weights; // [out_channels][in_channels][height][width]
    std::vector<float> bias;
    int in_channels, out_channels, kernel_size;

public:
    ConvolutionKernel(int in_ch, int out_ch, int k_size);

    void initializeWeights();

    // Getters
    const std::vector<std::vector<std::vector<std::vector<float>>>>& getWeights() const;
    const std::vector<float>& getBias() const;
    int getKernelSize() const;
    int getInChannels() const;
    int getOutChannels() const;
};

#endif // CONVOLUTION_KERNEL_H