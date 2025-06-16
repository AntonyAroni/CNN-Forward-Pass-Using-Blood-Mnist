//
// Created by antony on 6/15/25.
//

#include "ActivationFunction.h"
#include <algorithm>
#include <cmath>

float ActivationFunction::apply(float x, ActivationType type) {
    switch (type) {
        case ActivationType::RELU:
            return std::max(0.0f, x);
        case ActivationType::SIGMOID:
            return 1.0f / (1.0f + std::exp(-x));
        case ActivationType::TANH:
            return std::tanh(x);
        case ActivationType::LEAKY_RELU:
            return x > 0 ? x : 0.01f * x;
        default:
            return x;
    }
}

void ActivationFunction::applyToFeatureMap(std::vector<std::vector<std::vector<float>>>& feature_map, ActivationType type) {
    for (auto& channel : feature_map) {
        for (auto& row : channel) {
            for (auto& val : row) {
                val = apply(val, type);
            }
        }
    }
}
