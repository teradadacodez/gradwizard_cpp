This is a C++ program made by taking inspiration from the github repo of Andrej Karpathy, micrograd. This is C++ implementation of those concepts !
The main reason of using shared pointers in this implementation lies in the fact that the _backward function may be called much later after creation/declaration as it uses lambda, the reference much stay alive for that much time
