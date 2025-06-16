//
// Created by antony on 6/15/25.
//

#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include "Enums.h"
#include <vector>

class ActivationFunction {
public:
    static float apply(float x, ActivationType type);
    static void applyToFeatureMap(std::vector<std::vector<std::vector<float>>>& feature_map, ActivationType type);
};

#endif // ACTIVATION_FUNCTION_H