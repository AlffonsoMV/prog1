//
// Created by Alfonso Mateos on 28/12/23.
//

#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H

#include "layer.h"


class FullyConnectedLayer : public Layer {
private:
    std::vector<double> inputs;
    std::vector<double> outputs;
    std::vector<std::vector<double> > weights;
    std::vector<double> biases;
    std::vector<double> deltas; // For backpropagation
public:
    FullyConnectedLayer(int inputSize, int outputSize);
    void forward(const std::vector<double>& input);
    void computeDeltas(const std::vector<double>& nextDeltas, const std::vector<std::vector<double > > & nextWeights);

    std::vector<double> getInputs() { return inputs; }
    std::vector<double> getOutputs() { return outputs; }
    std::vector<std::vector<double > > getWeights() { return weights; }
    std::vector<double> getBiases() { return biases; }
    std::vector<double> getDeltas() { return deltas; }
};



#endif //FULLYCONNECTEDLAYER_H
