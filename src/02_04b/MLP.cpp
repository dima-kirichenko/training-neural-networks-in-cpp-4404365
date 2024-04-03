#include "MLP.h"

double frand(){
	return (2.0*(double)rand() / RAND_MAX) - 1.0;
}


// Return a new Perceptron object with the specified number of inputs (+1 for the bias).
Perceptron::Perceptron(size_t inputs, double bias){
	this->bias = bias;
	this->weights = std::vector<double>(inputs+1);
	std::generate(this->weights.begin(), this->weights.end(), frand);
}

// Run the perceptron. x is a vector with the input values.
double Perceptron::run(std::vector<double> x){
	x.push_back(this->bias);
	double sum = std::inner_product(x.begin(), x.end(), this->weights.begin(), 0.0);
	return this->sigmoid(sum);
}
