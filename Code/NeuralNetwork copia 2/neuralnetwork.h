#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Layer.h"
#include <vector>

class NeuralNetwork {
private:
    std::vector<Layer> layers;

public:
    NeuralNetwork() {};
    void addLayer(Layer l);
    void train(std::vector<std::pair<std::vector<double>, std::vector<double>>> dataset, int epochs, const double learningRate);
    std::vector<double> predict(const std::vector<double>& input);
};

#endif
