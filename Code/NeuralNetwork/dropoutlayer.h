#ifndef LAYERDROPOUT_H
#define LAYERDROPOUT_H

#include "layer.h"
#include <vector>

class DropoutLayer : public Layer
{
public:
    DropoutLayer(double dropoutRate);
    void forward(const std::vector<double>& input);
    void computeDeltas(const std::vector<double>& nextDeltas, const std::vector<std::vector<double>>& nextWeights);

private:
    double dropoutRate;
    std::vector<bool> dropoutMask;
};

#endif // LAYERDROPOUT_H
