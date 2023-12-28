#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cmath>

class Activation
{
public:
    virtual double f(double x) = 0; // Pure virtual function
    virtual double df(double x) = 0; // Pure virtual function
    virtual ~Activation() {} // Virtual destructor
};

class Sigmoid : public Activation
{
public:
    Sigmoid() {};
    double f(double x) override { return 1 / (1 + exp(-x)); }
    double df(double x) override { return f(x) * (1 - f(x)); }
};

#endif // ACTIVATION_H
