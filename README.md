# carnosaMNIST
CNN with evolution based training written from scratch
similar to a study conducted by Joseph Bruce from the University of Texas at Austin
https://nn.cs.utexas.edu/downloads/papers/bruce.gecco2001.pdf

main.cpp contains the evaluation and main loop function

net.cpp contains all network related functions 

## Version list
### v0.1 constant connection structure, constant connection size, 
- tested using around 15000 connections, inlcuding skipped connections

~25% accuracy
 
### v0.2 mutable connection Structure, constant connection size
- allowed for connections to be moved, but keeping same amount of connections

~35% accuracy
 
### v0.3 mutable connection structure, constant connection size
- Added convolution
- added weighted accuracy (more confident answers have higher weight)

~36% accuracy
 
### v0.4 fully connected (immutable) structure
- weight representation changed to matrix form
- added accelerated matrix multiplication (Apple Accelerate Library)
- added threaded training
- added biases
- removed weighted accuracy 

~75% accuracy with 784-200-10 CNN with 4x4 kernel and no pooling

### v0.45 fully connected, backpropogation inspired gene mixing
#### (scrapped, resulted in slower training)
- instead of modifying a random subset of genes
- every generation only a small subset of the weights can be modified
- similarly to backpropogation, where every weight is adjusted one at a time 
- however instead of taking the partial derivative of the cost function with respect to the weight
- we just change the weight by a random amount
- this should ideally prevent the network from getting stuck in local minima

<64% accuracy

### v0.5 fully connected, mixture of experts model
- joseph bruce paper

~?% accuracy

### v0.6 fully connected, backpropogation+evolution
- mix of back propogation and evolution based training
- every x generations, does reproduction, randomization, etc.
- between every reproduction cycle, uses gradient descent to adjust the weights and biases of the network

~?% accuracy

[![GitHub Trends SVG](https://api.githubtrends.io/user/svg/Stargor14/langs)](https://githubtrends.io)
