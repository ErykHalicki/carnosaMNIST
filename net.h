#include <stdlib.h>
#include<string>

#define inputSize 784//28*28 pixels
#define outputSize 10//10 digits
#define randomizationRate 0.1
static float randomizationPower=1;
#define input_width 28

/*
 deprecated data types

struct neuron{
    float value;
    short id;
};

struct connection{
    float weight;
    short src_id,dest_id, src_layer, dest_layer;//format (16 bits): src_layer(4 bits), src_id(12bits), copy for dest_layer and id (32 bits total)
    short src, dest;
    int full_id;
    //>>4 to get id, &15 to get layer
};
*/
class network{
    public:
    float **neurons, **kernel, **weights;
    int *layer_size; 
    int id,layers,neuronInLayer,connectionNum,kernel_size, stride, pooling_layer_size,convolution_layer_size;

    void init(int layers, int neuronInLayer, int k_s, int s);
    void clear();
    void run(unsigned char* input,float* result); 
    void randomize(float multiplier);
    void copy(network net);
    void serialize(std::string name);
    void read(std::string name);
    float inner_product(int x, int y);
    //float max_pool(int x, int y);
    void randomize_kernel(float multiplier);
};
void reproduce(network n1, network n2, network* offspring);
