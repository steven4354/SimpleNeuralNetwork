#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

class TrainingData
{
public:
	TrainingData(const string filename);
	bool isEof(void)
	{
		return m_trainingDataFile.eof();
	}
	void getTopology(vector<unsigned> &topology);

	// Returns the number of input values read from the file:
	unsigned getNextInputs(vector<double> &inputVals);
	unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
	ifstream m_trainingDataFile;
};

// Q: What's topology?
void TrainingData::getTopology(vector<unsigned> &topology)
{
	string line;
	string label;

	getline(m_trainingDataFile, line);
	stringstream ss(line);
	ss >> label;
	if(this->isEof() || label.compare("topology:") != 0)
	{
		abort();
	}

	while(!ss.eof())
	{
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}
	return;
}

TrainingData::TrainingData(const string filename)
{
	m_trainingDataFile.open(filename.c_str());
}


unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}

struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

// ****************** class Neuron ******************

class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVals);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
	vector<Connection> m_outputWeights;	// moved from private to public for logging
private:
	static double eta; // [0.0...1.0] overall net training rate
	static double alpha; // [0.0...n] multiplier of last weight change [momentum]
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	// randomWeight: 0 - 1
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double sumDOW(const Layer &nextLayer) const;
	double m_outputVal;
	unsigned m_myIndex;
	double m_gradient;
};

double Neuron::eta = 0.15; // overall net learning rate
double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight, [0.0..n]


void Neuron::updateInputWeights(Layer &prevLayer)
{
	// The weights to be updated are in the Connection container
	// in the nuerons in the preceding layer

	// Q: For the weights being updated, which "weights" are they? I thought each neuron has its own individualized weight of all past layer's neurons?
	// A: Yes, as you can see this class is for the Neuron class. So it's this specific Neuron's "idea" ("perception", "state", "set") of the weights of all past layer's Neurons
    //    As you can see each Neuron has a state of m_outputWeights (it's weight in the eyes of each subsequent Neuron)
	//    A weight for each m_myIndex Neuron (subsequent layer's Neuron)

	// Q: How is the m_myIndex set? Is it just increasing by 1? 
	//    e.g. Layer 1, Neron m_myIndex 1 | Layer 2, Neron m_myIndex 2, Layer 2, Neron m_myIndex 2 | Layer 3, Neron m_myIndex 3 ?

	// Q: How is the learning rate value determined?
	/* A: The learning rate is a hyperparameter, meaning it's set before the training process begins and typically requires some experimentation to get right. 
	      It controls how much we adjust the weights with respect to the gradient. Set it too low, and training will take forever. 
		  Set it too high, and you'll overshoot the optimal values, possibly never converging. 
		  There's no one-size-fits-all value; it's problem-dependent and requires some trial and error or more sophisticated techniques like learning rate schedules or adaptive learning rates.
	*/

	// Q: How is the alpha value determined?
	/* A: Alpha is the momentum coefficient, another hyperparameter. 
	      It helps the network to not get stuck in local minima and to smooth out the updates by considering the past weight changes. 
		  Like the learning rate, it's determined through experimentation. It's a balancing act; too much momentum and you might skip over minima, too little and you're not really benefiting from the concept. 
		  Again, there's no magic number; it's about trying different values and seeing what helps your network learn better.
	*/

	for(unsigned n = 0; n < prevLayer.size(); ++n)
	{
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight = 
				// Individual input, magnified by the gradient and train rate:
				eta
				* neuron.getOutputVal()
				// m_gradient state was updated by the previous steps in backProp
				* m_gradient
				// Also add momentum = a fraction of the previous delta weight
				+ alpha
				* oldDeltaWeight;
		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}
double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;

	// Sum our contributions of the errors at the nodes we feed

	// Q: How does this sumDow work? Explain the equation to me?
	
	// Q: It is iterating through each layer, but I thought m_outputWeights is for each neuron and not each layer?
	//    Why is n the layers and not the index of each neuron in the neural network?

	// Q: Why is it weight * error (gradient)?

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		// Print the m_outputWeights for debugging purposes
		// TODO: include which layer info as well
        std::cout << "m_outputWeights for Neuron index " << m_myIndex << " to Neuron index " << n << ": ";
        std::cout << "Weight: " << m_outputWeights[n].weight << ", ";
        std::cout << "DeltaWeight: " << m_outputWeights[n].deltaWeight << std::endl;

		// Actual logic
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	// Q: So this is calculating the m_gradient for "this" neuron and storing it in the state
	// Then this m_gradient is used to "update" the weights of all the past Neuron's in the eyes of this Neuron. (Each neuron weight is specific to the subsequent neuron)
	// Is my summary correct?

	// Q: m_outputVal was stored from feedForward correct?

	// Q: was m_outputVal created from after running it through activiation function (transfer function) or before? It's only considered "x" (an input) if before right?

	// Q: What does m_ stand for?
	// A: the prefix m_ is a naming convention used to indicate that a variable is a member of a class. It's a common practice in C++ (and other programming languages) to use a prefix or suffix to distinguish member variables (also known as instance variables or fields) from local variables and parameters.
	double dow = sumDOW(nextLayer); // dow stands for "derivative of weights"
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVals)
{
	double delta = targetVals - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

// tanh is used to produce "non-linearity" aka not a value between 0 and 1 (a line) but only 0 or 1 (a two cliffs)
double Neuron::transferFunction(double x)
{
	// tanh - output range [-1.0..1.0]
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
	// tanh derivative
	return 1.0 - x * x;
}

// feedForward basically just runs an input through the neuron to get the results
// both Neuron and Net class has a feedforward, Net essentially runs feedForward through all of its children Neurons
void Neuron::feedForward(const Layer &prevLayer)
{
	double sum = 0.0;

	// Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.

	for(unsigned n = 0 ; n < prevLayer.size(); ++n)
	{
		// Q: describe what prevLayer[n].m_outputWeights[m_myIndex].weight data structure looks like more intuitively. explain what m_myIndex is
		sum += prevLayer[n].getOutputVal() * 
				 prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	/*
	Q: Why is the neuron's output basically f(x) where x is the sum of the 
     previous layer's inputs and f is tanh?

	A: The neuron's output is calculated as f(x) to introduce non-linearity into 
		the network's operation. The sum of the previous layer's inputs (x) is 
		weighted and combined, and then an activation function f (in this case, 
		tanh) is applied. The tanh function squashes the input to a value between 
		-1 and 1. This non-linear transformation is essential because it allows 
		the neural network to learn and model complex patterns that are not 
		possible with linear operations alone.

	Q: Why is it the sum of the previous layers outputs * weight? Why not just sum? Or why not just activation function? Why is it essential to use the past layers outputs?

	A: The sum of the previous layer's outputs multiplied by their respective weights is a fundamental concept in neural networks. 
	   This operation, known as weighted sum, is essential because it combines the information from all the neurons in the previous layer in a meaningful way. 
	   Each neuron's output is considered in proportion to its weight, which represents the strength or importance of that connection. 
	   If we used just the sum without weights, we would lose the ability to differentiate the importance of different inputs. 
	   The activation function is then applied to this weighted sum to introduce non-linearity, which allows the network to learn and model complex patterns.
	   Without the non-linearity of the activation function, the network would be unable to learn anything beyond what a simple linear regression could provide.

	Q: Were the weights already applied? Why apply them again in each neuron?

	A: The weights are a crucial part of the neuron's computation. 
	   They are not applied beforehand but are used in real-time as the neuron processes its inputs. 
	   Each neuron in a layer takes the outputs from all neurons in the previous layer, multiplies each by the corresponding weight, and then sums these products to get the weighted sum. 
	   This weighted sum is then passed through the neuron's activation function to produce its output. 
	   The process is repeated for each neuron, with each neuron having its own set of weights for its inputs.

	Q: Why is the sum of the layer's output and not the previous neuron's output?

	A: In a neural network, each neuron in a layer receives inputs from all neurons in the previous layer, not just from a single neuron. 
	   This architecture allows the network to consider a wide range of features and patterns from the input data. 
	   The sum of the outputs of the entire previous layer, each weighted by the corresponding connection weight, provides a comprehensive set of inputs for each neuron to process.

	Q: Are all neurons just basically different activation functions? Are there 
		more "complex" types of neurons for other types of deep learning?

	A: While neurons can use a variety of activation functions (such as sigmoid, 
		ReLU, or tanh), the basic structure of a neuron is generally consistent: 
		they take inputs, apply weights, add a bias, and use an activation 
		function. However, there are more complex neuron-like structures in deep 
		learning, such as LSTM (Long Short-Term Memory) cells or GRU (Gated 
		Recurrent Unit) cells, which are used in recurrent neural networks (RNNs). 
		These structures have internal mechanisms designed to handle sequences and 
		memory, making them suitable for tasks like language modeling and time 
		series prediction.

	Q: Tell me how LSTM or GRU neurons differ from this
	*/
	m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for(unsigned c = 0; c < numOutputs; ++c){
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
}

// ****************** class Net ******************

class Net
{
public:
	Net(const vector<unsigned> &topology);
	void feedForward(const vector<double> &inputVals);
	void backProp(const vector<double> &targetVals);
	void getResults(vector<double> &resultVals) const;
	double getRecentAverageError(void) const { return m_recentAverageError; }

private:
	vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
	double m_error;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
};

double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

void Net::getResults(vector<double> &resultVals) const
{
	resultVals.clear();

	for(unsigned n = 0; n < m_layers.back().size() - 1; ++n)
	{
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

void Net::backProp(const std::vector<double> &targetVals)
{
	// Calculate overal net error (RMS of output neuron errors)
	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	// Q: Are the number of neurons for each layer hardcoded? How to "figure out" how many neurons to use?
	// A: Empirical magic

	// Q: What is the output layer?
	// A: The output layer is the last layer in the neural network where the final results are produced after processing the inputs through all the previous layers.
	for(unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		// Q: How does the target value correlate to the output layer? Is the output layer just the output?
		// A: The target value is what we expect the output layer to produce. Yes, the output layer is where the network's output values are.
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta *delta;
	}
	m_error /= outputLayer.size() - 1; // get average error squared
	
	// Q: How is the m_error used?
	// A: m_recentAverageError is used to smooth out the error over a number of training samples to see the trend, while m_error is the actual error for the current training sample.
	m_error = sqrt(m_error); // RMS

	// Implement a recent average measurement:

	// Q: Why is m_recentAverageError and m_error needed? Why not just m_error?
	m_recentAverageError = 
			(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
			/ (m_recentAverageSmoothingFactor + 1.0);
	
	// Calculate output layer gradients

	// Q: what is the calcOutputGradients doing in particular?
	// A: calcOutputGradients calculates the rate of change of the error with respect to the output neuron's output value, which is used to adjust the weights.

	// Q: calcOutputGradient changes a state variable in Neuron called m_gradient - how is that used?
	// A: m_gradient is used in the backpropagation process to determine how much the weights should be adjusted to reduce the error.

	/* Q: "m_gradient is used in the backpropagation process" explain the math that is doing this? and what variables it uses to do this?
	*  A: For an output neuron, the gradient is calculated as the product of the error delta (the difference between the neuron's output and the target value)
	*    and the derivative of the activation function with respect to the neuron's output.
	*
	*    This derivative provides a measure of how sensitive the neuron's
	*    output is to changes in its total input. 
	*
	*    Mathematically, for an output neuron, the gradient is:
	*    m_gradient = (targetVal - m_outputVal) * transferFunctionDerivative(m_outputVal).
	*
	*    For hidden neurons, the gradient is the dot product of the subsequent (following) layer's gradients and the neuron's outgoing weights (?), again scaled by the
	*    derivative of the activation function. This reflects the neuron's contribution to the error of neurons in the next layer. 
	* 
	*    The formula is:
	*    m_gradient = sumDOW(nextLayer) * transferFunctionDerivative(m_outputVal).
	*
	*    Here, sumDOW is a helper function that computes the sum of the product of weights and gradients of the next layer's neurons.
	*    The variables involved are the neuron's output (m_outputVal), the target output (targetVal), and the neuron's weights (m_outputWeights).
	*
	*  Q: (sub-question) Why is the gradient (measure of error) not just the target output minus actual output, but instead it's multiplied by derivative of actvivation func?
	*  A: The derivative of the activation function tells us how sensitive the output of a neuron is to changes in its weighted input. 
	*     By multiplying the error (target minus actual output) by the derivative of the activation function, we're scaling the error by this sensitivity, which gives us the gradient of the error with respect to the weights.
	*
	*  Q: (sub-question) Why is the derivative of the activation function used and not the activation function itself to calculate the gradient?
	*  A: See question before this, it allows us to scale the error/gradient to an appropriate change in weight
	*/

	/* Q: What's the activation function formula? What line of code shows the activation function?
	*  A: The activation function used in this neural network is the hyperbolic tangent function (tanh), which is mathematically represented as:
	*    tanh(x) = (e^x - e^-x) / (e^x + e^-x).
	*
	*    The line of code that implements the activation function is:
	*    `return tanh(x);` on line 189
	* 
	*    This line is part of the `transferFunction` method within the `Neuron` class. The derivative of this function, which is crucial for backpropagation,
	*    is given by:
	*    `return 1.0 - x * x;`
	*
	*    This line is part of the `transferFunctionDerivative` method and calculates the derivative of the tanh function, which is 1 - tanh^2(x).
	*
	*  Q: (sub-question) How do you figure out the derivative of (e^x - e^-x) / (e^x + e^-x)? What's the calculus rules for this again? 
	*  A: Calculus: Quotient rule then chain rule
	*/

	// Q: what is the relationship between Neuron and m_layers?
	// A: m_layers is a collection of layers, and each layer is a collection of Neuron objects. Neurons are the basic units that make up each layer.

	// Q: how does the code choose how many neurons each layer should have?
	// A: The number of neurons in each layer is determined by the network's topology, which is a design choice made by the programmer or derived from the problem's requirements. It's not something the code figures out on its own; it's specified in the training data file under the "topology:" section. You need to experiment with different topologies to find the one that works best for your specific problem.

	// Q: in each layer are neurons adjacent to each other (no data exchanged between neurons), or are they interleaved (neurons within a layer exchanges data between each other)?
	// A: Neurons within the same layer do not exchange data with each other. Each neuron in a layer processes its inputs independently and passes its output to the next layer. The actual "communication" happens between layers, not within them.
	for(unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}
	
	// Calculate gradients on hidden layers

	// Q: Again once m_gradient is updated by calcHiddenGradients, how is it used?
	// A: Once updated, m_gradient is used in the weight update step to adjust the weights of the neurons in the hidden layers to reduce the error.

	// Q: Give me the math formula for the above Q/A
	// A: m_outputWeights[n].weight * nextLayer[n].m_gradient * activation function derivative

	for(unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
	{
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for(unsigned n = 0; n < hiddenLayer.size(); ++n)
		{
			// Debug output for layer information
            std::cout << "Calculating gradients for Layer " << layerNum
                      << ", Neuron index " << n << std::endl;

			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// For all layers from outputs to first hidden layer,
	// update connection weights

	// Q: what's the difference between output layers and hidden layers?
	// A: The output layer is the final layer that produces the network's output. Hidden layers are all the layers between the input and output layers that process the inputs.

	for(unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for(unsigned n = 0; n < layer.size() - 1; ++n)
		{
			// Q: How does it update the weights using the gradient?
			// The code uses the gradient to calculate the change in weights (delta weights) by:
			// gradient * learning rate (eta) + momentum (alpha times the previous delta weight). 
			// This calculated delta weight is then added to the current weight to adjust it. 
			// The idea is to reduce the error by nudging the weights in the direction that decreases the gradient of the error with respect to the weights.

			// Q: What's "input" weights? What does the "input" mean?
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::feedForward(const vector<double> &inputVals)
{
	// Check the num of inputVals euqal to neuronnum expect bias
	assert(inputVals.size() == m_layers[0].size() - 1);

	// Assign {latch} the input values into the input neurons
	for(unsigned i = 0; i < inputVals.size(); ++i){
		m_layers[0][i].setOutputVal(inputVals[i]); 
	}

	// Forward propagate
	for(unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum){
		Layer &prevLayer = m_layers[layerNum - 1];
		for(unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n){
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

Net::Net(const vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
		m_layers.push_back(Layer());
		// numOutputs of layer[i] is the numInputs of layer[i+1]
		// numOutputs of last layer is 0
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 :topology[layerNum + 1];

		// We have made a new Layer, now fill it ith neurons, and
		// add a bias neuron to the layer:
		for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			// cout << "Made a Neuron!" << endl;
			// Improved:
			cout << "Made a Neuron in Layer " << layerNum << " with index " << neuronNum << " and " << numOutputs << " outputs." << endl;
		}

		// Force the bias node's output value to 1.0. It's the last neuron created above
		m_layers.back().back().setOutputVal(1.0);
	}
}

void showVectorVals(string label, vector<double> &v)
{
	cout << label << " ";
	for(unsigned i = 0; i < v.size(); ++i)
	{
		cout << v[i] << " ";
	}
	cout << endl;
}

// IMPORTANT: this can be removed, set low for learning purposes to see how backProp is functioning
int total_passes_allowed = 2;

int main()
{
	TrainingData trainData("trainingData.txt");
	//e.g., {3, 2, 1 }
	vector<unsigned> topology;

	// Q: what's push back doing? and why is it commented out?
	//topology.push_back(3);
	//topology.push_back(2);
	//topology.push_back(1);

	trainData.getTopology(topology);
	Net myNet(topology);

	vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;
	while(!trainData.isEof() && trainingPass < total_passes_allowed)
	{
		++trainingPass;
		cout << endl << "Pass" << trainingPass;

		// Get new input data and feed it forward:
		if(trainData.getNextInputs(inputVals) != topology[0])
			break;
		showVectorVals(": Inputs :", inputVals);
		myNet.feedForward(inputVals);

		// Collect the net's actual results:
		myNet.getResults(resultVals);
		showVectorVals("Outputs:", resultVals);

		// Train the net what the outputs should have been:
		trainData.getTargetOutputs(targetVals);
		showVectorVals("Targets:", targetVals);
		assert(targetVals.size() == topology.back());

		// Q: How often is backProp used? Is it reused on each input?
		/*
		   A: The backProp function is called once for every individual training example during the training process. In the context of neural networks, a "training example" refers to a single instance of input data and its corresponding target output.

              During each epoch (a full pass through the entire training dataset), backProp is used to update the weights after each forward pass of a training example. This means that if you have a dataset with 1,000 training examples, and you run the training for 10 epochs, backProp will be called 10,000 times – once for each training example in each epoch.

              The purpose of reusing backProp on each input is to iteratively adjust the network's weights in response to the error observed for each training example. This is how the network learns from the data over time.
		*/

		// Q: Will backprop eventually produce no further improvements in the error? Do we always want to run backprop to the point that error is zero? Is a zero error even possible?
		// A: Yes, if it produces zero error/gradient it usually means the neural network "overfit" the training set and may perform poorly on new data
		myNet.backProp(targetVals);

		// Report how well the training is working, average over recnet
		cout << "Net recent average error: "
		     << myNet.getRecentAverageError() << endl;
	}

	cout << endl << "Done" << endl;

}
