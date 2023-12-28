#include "neuralnetwork.h"
#include "Layer.h"
#include <iostream>
#include <vector>

int main() {
    NeuralNetwork nn;

    Layer ln1(3,4);
    nn.addLayer(ln1);

    Layer ln2(4,5);
    nn.addLayer(ln2);

    Layer ln3(5,1);
    nn.addLayer(ln3);

    std::vector<double> input = {1.0, 0.0, 1.0}; // Example input
    std::vector<double> target = {1.0}; // Example target

    nn.train(input, target, 100, 0.01);

    auto output = nn.predict(input);
    std::cout << "Predicted output: " << output[0] << std::endl;

    return 0;
}
