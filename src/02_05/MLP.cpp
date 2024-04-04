#include "MLP.h"

double frand(){
	return (2.0*(double)rand() / RAND_MAX) - 1.0;
}


// Return a new Perceptron object with the specified number of inputs (+1 for the bias).
Perceptron::Perceptron(size_t inputs, double bias){
	this->bias = bias;
	weights.resize(inputs+1);
	generate(weights.begin(),weights.end(),frand);
}

// Run the perceptron. x is a vector with the input values.
double Perceptron::run(std::vector<double> x){
	x.push_back(bias);
	double sum = inner_product(x.begin(),x.end(),weights.begin(),(double)0.0);
	return sigmoid(sum);
}

// Challenge: Finish the following functions:

void Perceptron::set_weights(std::vector<double> w_init){
	// w_init is a vector of doubles. Organize it as you'd like.
	// Set the weights of the perceptron to the values in w_init.
	// Validate that the size of w_init is the same as the size of the weights vector.
	// If the sizes don't match, throw an exception.
	// If the sizes do match, set the weights to the values in w_init.
	if(w_init.size() != weights.size()){
		throw std::invalid_argument("The size of the input vector does not match the size of the weights vector.");
	}
	weights = w_init;
}

double Perceptron::sigmoid(double x){
	// Return the output of the sigmoid function applied to x
	return 1.0 / (1.0 + exp(-x));
}
