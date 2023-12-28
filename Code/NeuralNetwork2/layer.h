#ifndef LAYER_H
#define LAYER_H

#include <vector>

class Layer {
public:
    virtual ~Layer() {}
    virtual void forward(const std::vector<double>& input) = 0;
    virtual void computeDeltas(const std::vector<double>& nextDeltas, const std::vector<std::vector<double> > & nextWeights) = 0;

    virtual std::vector<double> getInputs() = 0;
    virtual std::vector<double> getOutputs() = 0;
    virtual std::vector<std::vector<double> > getWeights() = 0;
    virtual std::vector<double> getBiases() = 0;
    virtual std::vector<double> getDeltas() = 0;
};

#endif
