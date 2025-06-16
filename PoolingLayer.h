//
// Created by antony on 6/15/25.
//

#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "Enums.h"
#include <vector>

class PoolingLayer {
private:
    int pool_size;
    int stride;
    PoolingType type;

public:
    PoolingLayer(int p_size, int s, PoolingType t);

    std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<std::vector<float>>>& input);
};

#endif // POOLING_LAYER_H