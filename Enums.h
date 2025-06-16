//
// Created by antony on 6/15/25.
//

#ifndef ENUMS_H
#define ENUMS_H

// Enum para tipos de pooling
enum class PoolingType {
    MAX,
    MIN,
    AVERAGE
};

// Enum para funciones de activación
enum class ActivationType {
    RELU,
    SIGMOID,
    TANH,
    LEAKY_RELU
};

#endif // ENUMS_H