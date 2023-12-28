#ifndef LAYERFULLYCONNECTED_H
#define LAYERFULLYCONNECTED_H


#include "activation.h"
#include <vector>
#include "Layer.h"

class LayerFullyConnected : public Layer
{
public:
    LayerFullyConnected(int inputSize, int outputSize, Activation* act);
    void forward(const std::vector<double>& input);
    void computeDeltas(const std::vector<double>& nextDeltas, const std::vector<std::vector<double>>& nextWeights);
};

#endif // LAYERFULLYCONNECTED_H
