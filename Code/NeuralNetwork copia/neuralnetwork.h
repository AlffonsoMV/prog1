#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Layer.h"
#include <vector>

class NeuralNetwork {
private:
    std::vector<Layer> layers;

public:
    NeuralNetwork() {};
    NeuralNetwork(const std::vector<int>& layerSizes);
    void addLayer(Layer l);
    void addLayer(int inputSize, int outputSize);
    void train(const std::vector<double>& input, const std::vector<double>& target, int epochs, const double learningRate);
    std::vector<double> predict(const std::vector<double>& input);
};

#endif
