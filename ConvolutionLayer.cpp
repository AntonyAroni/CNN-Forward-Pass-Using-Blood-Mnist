//
// Created by antony on 6/15/25.
//

#include "ConvolutionLayer.h"
#include <iostream>

ConvolutionLayer::ConvolutionLayer(int in_channels, int out_channels, int kernel_size,
                                   ActivationType act_type, int s, int p)
    : kernel(in_channels, out_channels, kernel_size), activation_type(act_type), stride(s), padding(p) {}

std::vector<std::vector<std::vector<float>>> ConvolutionLayer::forward(const std::vector<std::vector<std::vector<float>>>& input) {
    int in_channels = input.size();
    int input_height = input[0].size();
    int input_width = input[0][0].size();

    int output_height = (input_height + 2 * padding - kernel.getKernelSize()) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel.getKernelSize()) / stride + 1;

    // Verificar que las dimensiones de salida sean válidas
    if (output_height <= 0 || output_width <= 0) {
        std::cerr << "Error: Dimensiones de salida inválidas. Input: " << input_height << "x" << input_width
                  << ", Kernel: " << kernel.getKernelSize() << "x" << kernel.getKernelSize()
                  << ", Output calculado: " << output_height << "x" << output_width << std::endl;
        // Retornar la entrada sin procesar en caso de error
        return input;
    }

    std::vector<std::vector<std::vector<float>>> output(kernel.getOutChannels(),
        std::vector<std::vector<float>>(output_height, std::vector<float>(output_width, 0.0f)));

    const auto& weights = kernel.getWeights();
    const auto& bias = kernel.getBias();

    // Operación de convolución
    for (int out_ch = 0; out_ch < kernel.getOutChannels(); out_ch++) {
        for (int i = 0; i < output_height; i++) {
            for (int j = 0; j < output_width; j++) {
                float sum = bias[out_ch];

                for (int in_ch = 0; in_ch < in_channels; in_ch++) {
                    for (int ki = 0; ki < kernel.getKernelSize(); ki++) {
                        for (int kj = 0; kj < kernel.getKernelSize(); kj++) {
                            int input_i = i * stride - padding + ki;
                            int input_j = j * stride - padding + kj;

                            if (input_i >= 0 && input_i < input_height &&
                                input_j >= 0 && input_j < input_width) {
                                sum += input[in_ch][input_i][input_j] * weights[out_ch][in_ch][ki][kj];
                            }
                        }
                    }
                }

                output[out_ch][i][j] = sum;
            }
        }
    }

    // Aplicar función de activación
    ActivationFunction::applyToFeatureMap(output, activation_type);

    return output;
}

void ConvolutionLayer::printKernelInfo() const {
    std::cout << "Kernel: " << kernel.getInChannels() << " -> " << kernel.getOutChannels()
              << " channels, size: " << kernel.getKernelSize() << "x" << kernel.getKernelSize() << std::endl;
}
