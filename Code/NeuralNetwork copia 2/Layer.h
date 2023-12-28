#ifndef LAYER_H
#define LAYER_H

#include "activation.h"
#include <vector>

class Layer {
public:
    std::vector<double> inputs;
    std::vector<double> outputs;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> deltas; // For backpropagation
    Activation* act;

    Layer(int inputSize, int outputSize, Activation* act);
    void forward(const std::vector<double>& input);
    void computeDeltas(const std::vector<double>& nextDeltas, const std::vector<std::vector<double>>& nextWeights);
};

#endif
