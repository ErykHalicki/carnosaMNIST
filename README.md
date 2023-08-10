# carnosaMNIST
CNN with evolution based training written from scratch

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

~52.75% accuracy

### v0.5 fully connected, backpropogation + evolution
- mix of back propogation and evolution based training
- every x generations, does reproduction, randomization, etc.
- between every reproduction cycle, uses gradient descent to adjust the weights and biases of the network

~?% accuracy
