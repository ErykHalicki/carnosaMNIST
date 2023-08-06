#include <stdlib.h>
#include<string>
#include <unordered_set>

#define inputSize 784//28*28 pixels
#define outputSize 10//10 digits
#define randomizationRate 0.01
#define randomizationStrength 1
#define input_width 28

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

class network{
    public:
    neuron** neurons;
    connection* connections;
    int id,layers,neuronInLayer,connectionNum,kernel_size, stride, convolution_layer_size;
    long max_connections;
    //stride is how far the kernel shifts every iteration
    float** kernel;//kernel is a square, of size [k_s*k_s]
    std::unordered_set<int>* connection_ids;

    void init(int layers, int neuronInLayer, int randomConnections, int k_s, int s);
    //connections only created once, later generations just change the weights
    void clear();
    void sort();
    void run(unsigned char* input,float* result); 
    void randomize(float multiplier);
    void copy(network net);
    void serialize(std::string name);
    void read(std::string name);
    void destroy();
    void add_connection(int amount);
    void rearrange_connection(int amount);
    void reinforce(int result, float multiplier);
    float inner_product(int x, int y);
    void randomize_kernel(float multiplier);
    network();
};
network* mix(network n1, network n2);
void reproduce(network n1, network n2, network* offspring);
short create_id(int layer, int id);
void create_full_id(connection* c);
