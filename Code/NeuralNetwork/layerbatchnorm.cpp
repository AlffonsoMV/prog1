#include "layerbatchnorm.h"
#include <numeric>
#include <cmath>

LayerBatchNorm::LayerBatchNorm(int size) : inputSize(size), gamma(size, 1.0), beta(size, 0.0) {}

void LayerBatchNorm::forward(const std::vector<double>& input) {
    mean.clear();
    variance.clear();
    outputs.resize(inputSize);

    for (int i = 0; i < inputSize; ++i) {
        mean.push_back(std::accumulate(input.begin(), input.end(), 0.0) / inputSize);
    }

    for (int i = 0; i < inputSize; ++i) {
        double var = 0.0;
        for (double val : input) {
            var += std::pow(val - mean[i], 2);
        }
        variance.push_back(var / inputSize);
    }

    for (int i = 0; i < inputSize; ++i) {
        outputs[i] = gamma[i] * (input[i] - mean[i]) / std::sqrt(variance[i] + 1e-8) + beta[i];
    }
}

void LayerBatchNorm::computeDeltas(const std::vector<double>& nextDeltas, const std::vector<std::vector<double>>& nextWeights) {
    deltas.resize(inputSize);
    for (int i = 0; i < inputSize; ++i) {
        deltas[i] = nextDeltas[i] * gamma[i] / std::sqrt(variance[i] + 1e-8);
    }
}

void LayerBatchNorm::updateParameters(double learningRate) {
    for (int i = 0; i < inputSize; ++i) {
        gamma[i] -= learningRate * deltas[i] * (outputs[i] - beta[i]) / std::sqrt(variance[i] + 1e-8);
        beta[i] -= learningRate * deltas[i];
    }
}
