#ifndef LAYERBATCHNORM_H
#define LAYERBATCHNORM_H

#include "Layer.h"
#include <vector>

class LayerBatchNorm : public Layer
{
public:
    LayerBatchNorm(int inputSize);
    void forward(const std::vector<double>& input);
    void computeDeltas(const std::vector<double>& nextDeltas, const std::vector<std::vector<double>>& nextWeights);
    void updateParameters(double learningRate);

private:
    int inputSize;
    std::vector<double> gamma; // Scale parameters
    std::vector<double> beta;  // Shift parameters
    std::vector<double> mean;  // Mean of the batch
    std::vector<double> variance; // Variance of the batch
};

#endif // LAYERBATCHNORM_H
