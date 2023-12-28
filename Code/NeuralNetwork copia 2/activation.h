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

class Identity : public Activation
{
public:
    Identity() {};
    double f(double x) override { return x; }
    double df(double x) override { return 1; }
};

class Sigmoid : public Activation
{
public:
    Sigmoid() {};
    double f(double x) override { return 1 / (1 + exp(-x)); }
    double df(double x) override { return f(x) * (1 - f(x)); }
};

class TanH : public Activation
{
public:
    TanH() {};
    double f(double x) override { return 2 / (1 + exp(-2*x)) - 1; }
    double df(double x) override { return 1 - f(x)*f(x); }
};

class ReLu : public Activation
{
public:
    ReLu() {};
    double f(double x) override { return (x < 0 ? 0.0 : x); }
    double df(double x) override { return (x < 0 ? 0.0 : 1.0); }
};

#endif // ACTIVATION_H
