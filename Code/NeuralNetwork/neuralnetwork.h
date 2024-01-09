#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "layer.h"
#include <vector>
#include <utility>

enum class TrainingMode {
    BATCH,
    STOCHASTIC
};

class NeuralNetwork {
public:
    void addLayer(Layer* l);
    void train(std::vector<std::pair<std::vector<double>, std::vector<double>>> dataset, int epochs, TrainingMode mode, double learningRate = 0.01);
    std::vector<double> predict(const std::vector<double>& input);
    ~NeuralNetwork();

private:
    std::vector<Layer*> layers;
    std::vector<std::vector<std::vector<double>>> accumulatedWeightGradients;
    std::vector<std::vector<double>> accumulatedBiasGradients;

    void updateWeightsAndBiases(const std::vector<double>& input, double learningRate);
    void initializeGradientAccumulators();
    void accumulateGradients();
    void applyAccumulatedGradients(double learningRate);
};

#endif // NEURALNETWORK_H
