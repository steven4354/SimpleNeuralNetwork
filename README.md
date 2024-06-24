# SimpleNeuralNetwork

This is a C++ implement of simple neural network. It's based on video [Neural Net in C++ Tutorial](https://vimeo.com/19569529) by David Miller.

# Explanation of XOR

XOR (exclusive OR) returns true (1) if exactly one of the inputs is true, and false (0) otherwise.

Further info: https://www.perplexity.ai/search/whats-the-xor-wqWWSOp3QxCWlBCjnoeVLA

# Test in Ubuntu

1 Gernerate training data to slove XOR problem

```bash
    cd src
    g++ ./makeTrainingSamples.cpp -o makeTrainingSamples
    ./makeTrainingSamples > out.txt
```
2 Test neural netwrok

```bash
    g++ ./neural-net.cpp -o neural-net
    ./neural-net
```

And you will get the result!

Example:

```bash
Pass1992: Inputs : 1 1
Outputs: -0.00151756
Targets: 0
Net recent average error: 0.0209756

Pass1993: Inputs : 0 1
Outputs: 0.965941
Targets: 1
Net recent average error: 0.0211051

Pass1994: Inputs : 1 1
Outputs: 0.00285765
Targets: 0
Net recent average error: 0.0209245

Pass1995: Inputs : 1 1
Outputs: 0.00141137
Targets: 0
Net recent average error: 0.0207313

Pass1996: Inputs : 1 0
Outputs: 0.967809
Targets: 1
Net recent average error: 0.0208447

Pass1997: Inputs : 1 0
Outputs: 0.967882
Targets: 1
Net recent average error: 0.0209563

Pass1998: Inputs : 1 1
Outputs: 0.00107933
Targets: 0
Net recent average error: 0.0207595

Pass1999: Inputs : 0 1
Outputs: 0.966028
Targets: 1
Net recent average error: 0.0208904

Pass2000: Inputs : 1 1
Outputs: 0.00208086
Targets: 0
Net recent average error: 0.0207041

Pass2001: Inputs : 0 0
Outputs: 0.00246967
Targets: 0
Net recent average error: 0.0205236

Pass2002
Done
```

# TODO

Rework the code to add another hidden layer to understand how the m_outputWeights[n].weight for each Neuron might look if multiple layers uses its output. (Or perhaps just subsequent layer only does - need to know)
