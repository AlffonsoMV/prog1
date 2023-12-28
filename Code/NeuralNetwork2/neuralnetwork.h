#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "layer.h"
#include <vector>

class NeuralNetwork {
private:
    std::vector<std::unique_ptr<Layer> > layers;
public:
    enum TrainingMode {
        BATCH,
        STOCHASTIC
    };
    NeuralNetwork();
    void addLayer(Layer* layer);
    void train(const std::vector<std::pair<std::vector<double>, std::vector<double> > > & dataset, int epochs, double learningRate = 0.01, TrainingMode mode = BATCH);
    std::vector<double> predict(const std::vector<double> & input);
    void forwardBackwardPass(const std::vector<double> & input, const std::vector<double> & target);
    void updateWeightsBiases(std::vector<std::vector<double> > & weightGradients, std::vector<std::vector<double> > & biasGradients, double learningRate, size_t batchSize);
    void updateWeightsBiasesStochastic(const std::vector<double> & input, double learningRate);
};

#endif
