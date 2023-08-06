#include <stdlib.h>
#include<string>
#define inputSize 784//28*28 pixels
#define outputSize 10//10 digits
#define randomizationRate 0.01
#define randomizationStrength 5

struct neuron{
    float value;
    short id;
};

struct connection{
    float weight;
    short src,dest;//format (16 bits): src_layer(4 bits), src_id(12bits), copy for dest_layer and id (32 bits total)
    //>>4 to get id, &15 to get layer
};

class network{
    public:
    neuron** neurons;
    connection* connections;
    int id,layers,neuronInLayer,connectionNum;//id used for training purposes 

    void init(int layers, int neuronInLayer, int randomConnections);
    //connections only created once, later generations just change the weights
    void clear();
    void sort();
    void run(unsigned char* input,float* result); 
    void randomize();
    void copy(network net);
    void serialize(std::string name);
    void read(std::string name);
    void destroy();
    network();
};
network* mix(network n1, network n2);
void reproduce(network n1, network n2, network* offspring);
short createID(int layer, int id);
