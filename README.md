# carnosaMNIST
    CNN with evolution based training written from scratch

## Version list
 * v0.1 - constant connection structure, constant connection size, 

 *      ~25% accuracy
 
 * v0.2 - mutable connection Structure, constant connection size 

 *      ~35% accuracy
 
 * v0.3 - mutable connection structure, constant connection size
 *      - Added convolution
 *      - added weighted accuracy (more confident answers have higher weight)

 *      ~36% accuracy
 
 * v0.4 - fully connected (immutable) structure
 *      - weight representation changed to matrix form
 *      - added accelerated matrix multiplication (Apple Accelerate Library)
 *      - added threaded training
 *      - added biases
 *      - removed weighted accuracy
 
 *      ~51% accuracy
