//
// Created by antony on 6/15/25.
//

#include "PoolingLayer.h"
#include <algorithm>

PoolingLayer::PoolingLayer(int p_size, int s, PoolingType t)
    : pool_size(p_size), stride(s), type(t) {}

std::vector<std::vector<std::vector<float>>> PoolingLayer::forward(const std::vector<std::vector<std::vector<float>>>& input) {
    int channels = input.size();
    int input_height = input[0].size();
    int input_width = input[0][0].size();

    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;

    std::vector<std::vector<std::vector<float>>> output(channels,
        std::vector<std::vector<float>>(output_height, std::vector<float>(output_width, 0.0f)));

    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < output_height; i++) {
            for (int j = 0; j < output_width; j++) {
                float result = 0.0f;
                bool first = true;

                for (int pi = 0; pi < pool_size; pi++) {
                    for (int pj = 0; pj < pool_size; pj++) {
                        int input_i = i * stride + pi;
                        int input_j = j * stride + pj;

                        if (input_i < input_height && input_j < input_width) {
                            float val = input[c][input_i][input_j];

                            if (first) {
                                result = val;
                                first = false;
                            } else {
                                switch (type) {
                                    case PoolingType::MAX:
                                        result = std::max(result, val);
                                        break;
                                    case PoolingType::MIN:
                                        result = std::min(result, val);
                                        break;
                                    case PoolingType::AVERAGE:
                                        result += val;
                                        break;
                                }
                            }
                        }
                    }
                }

                if (type == PoolingType::AVERAGE) {
                    result /= (pool_size * pool_size);
                }

                output[c][i][j] = result;
            }
        }
    }

    return output;
}