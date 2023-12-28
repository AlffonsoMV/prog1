#include "dropoutlayer.h"
#include <random>

DropoutLayer::DropoutLayer(double rate) : dropoutRate(rate) {}

void DropoutLayer::forward(const std::vector<double>& input) {
    dropoutMask.resize(input.size());
    outputs.resize(input.size());

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (size_t i = 0; i < input.size(); ++i) {
        dropoutMask[i] = (distribution(generator) > dropoutRate);
        outputs[i] = dropoutMask[i] ? input[i] : 0.0;
    }
}

void DropoutLayer::computeDeltas(const std::vector<double>& nextDeltas, const std::vector<std::vector<double>>& nextWeights) {
    deltas.resize(nextDeltas.size());
    for (size_t i = 0; i < nextDeltas.size(); ++i) {
        deltas[i] = dropoutMask[i] ? nextDeltas[i] : 0.0;
    }
}

