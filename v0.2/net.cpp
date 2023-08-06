#include "net.h"
#include <iostream>
#include <time.h>
#include <cmath>

short createID(int layer, int id){
    short res=0;
    res|=id;
    res<<=4;
    res|=layer;
    return res; 
}
void network::sort(){
    for(int i = 1;i < connectionNum; i++) {
        int j = i;
        while(j > 0 && (connections[j].src_layer) < (connections[j-1].src_layer)) {
            connection temp = connections[j];
            connections[j] = connections[j-1];
            connections[j-1] = temp;
            j--;
        }
    }
}
network::network(){
}
void network::init(int l, int ninl,int randomConnections){
    layers=l;
    neuronInLayer=ninl;
    neurons=(neuron**)malloc(sizeof(neuron*)*layers);
    connectionNum=randomConnections;
    for(int i=0;i<layers;i++){
        if(i==0)neurons[i]=(neuron*)malloc(sizeof(neuron)*inputSize);
        else if(i==layers-1)neurons[i]=(neuron*)malloc(sizeof(neuron)*outputSize);
        else neurons[i]=(neuron*)malloc(sizeof(neuron)*neuronInLayer);
        for(int j=0;j<neuronInLayer;j++){
            neurons[i][j].id=createID(i,j);   
            neurons[i][j].value=0;
        }
    }

    connections=(connection*)malloc(sizeof(connection)*randomConnections);

    short src_id,src_layer,dest_id,dest_layer;
    for(int i=0;i<randomConnections;i++){
        connections[i].weight=(rand()%200-100)/100.0;//-1.0 - 1.0 

        src_layer=rand()%(layers-1);//cant be last layer
        if(src_layer==0) src_id=rand()%inputSize;
        else src_id=rand()%neuronInLayer;

        dest_layer=rand()%(layers-1)+1;//cant be first layer
        if(dest_layer==layers-1) dest_id=rand()%outputSize;//0-9 
        else dest_id=rand()%neuronInLayer;

        connections[i].src_layer=src_layer;
        connections[i].src_id=src_id;
        connections[i].dest_layer=dest_layer;
        connections[i].dest_id=dest_id;
    }
    sort();
}
void network::clear(){
    int temp=inputSize;
    for(int i=0;i<layers;i++){
        if(i>0)temp=neuronInLayer;
        if(i==layers-1)temp=outputSize;
        for(int j=0;j<temp;j++){
            neurons[i][j].value=0;
        }
    }
}

void network::run(unsigned char* input, float* result){
    //loading input values to layer 0 
    for(int i=0;i<inputSize;i++){
        neurons[0][i].value=std::min(255,(int)input[i])/255.0;
    }
    for(int i=0;i<connectionNum;i++){
        /*
        src_id=connections[i].src>>4;
        src_layer=connections[i].src&15;
        dest_id=connections[i].dest>>4;
        dest_layer=connections[i].dest&15;
        weight=connections[i].weight;
        srcValue=neurons[src_layer][src_id].value;//relu activtion function
        */
        //speed up functioin, maybe run profiler
        neurons[connections[i].dest_layer][connections[i].dest_id].value+=std::fmax(0,neurons[connections[i].src_layer][connections[i].src_id].value*connections[i].weight);
    }
    /*
    float sum=0;
    for(int i=0;i<outputSize;i++){
        sum+=neurons[layers-1][i].value;
    }
    
    */
    for(int i=0;i<outputSize;i++){
        result[i]=neurons[layers-1][i].value;
    }
    clear();
}
void network::randomize(){
    int rs= randomizationStrength*100;
   // srand (time(NULL));
    for(int i=0;i<connectionNum;i++){
        if((rand()%100+1)/100.0 <= randomizationRate)
            connections[i].weight+=rand()%((rs*2+1)-rs)/100.0;//from -rs to +rs
    }
}
void reproduce(network n1, network n2,network* offspring){
    int halfConnections=rand()%n1.connectionNum/2, allConnections=std::max(n1.connectionNum,n2.connectionNum);

    if(offspring[0].connectionNum<n2.connectionNum){
        offspring[0].connections=(connection*)realloc(offspring[0].connections,sizeof(connection)*n2.connectionNum);
        offspring[0].connectionNum=n2.connectionNum;
    }
    if(offspring[1].connectionNum<n1.connectionNum){
        offspring[1].connections=(connection*)realloc(offspring[1].connections,sizeof(connection)*n1.connectionNum);
        offspring[1].connectionNum=n1.connectionNum;
    }
    if(offspring[3].connectionNum<n2.connectionNum){
        offspring[3].connections=(connection*)realloc(offspring[3].connections,sizeof(connection)*n2.connectionNum);
        offspring[3].connectionNum=n2.connectionNum;
    }
    if(offspring[2].connectionNum<n1.connectionNum){
        offspring[2].connections=(connection*)realloc(offspring[2].connections,sizeof(connection)*n1.connectionNum);
        offspring[2].connectionNum=n1.connectionNum;
    }
    memcpy(offspring[0].connections,n1.connections,sizeof(connection)*allConnections);
    memcpy(offspring[1].connections,n2.connections,sizeof(connection)*allConnections);
    memcpy(&offspring[0].connections[halfConnections],&n2.connections[halfConnections],sizeof(connection)*(n2.connectionNum-halfConnections)); 
    memcpy(&offspring[1].connections[halfConnections],&n1.connections[halfConnections],sizeof(connection)*(n1.connectionNum-halfConnections)); 
    memcpy(offspring[2].connections,n1.connections,sizeof(connection)*n1.connectionNum);
    memcpy(offspring[3].connections,n2.connections,sizeof(connection)*n2.connectionNum);

    for(int i=2;i<4;i++)
        for(int j=0;j<100;j++)
        offspring[i].randomize();
}
void network::copy(network net){
    layers=net.layers;
    neuronInLayer=net.neuronInLayer;
    neurons=(neuron**)malloc(sizeof(neuron*)*layers);
    connectionNum=net.connectionNum;
    for(int i=0;i<layers;i++){
        if(i==0)neurons[i]=(neuron*)malloc(sizeof(neuron)*inputSize);
        else if(i==layers-1)neurons[i]=(neuron*)malloc(sizeof(neuron)*outputSize);
        else neurons[i]=(neuron*)malloc(sizeof(neuron)*neuronInLayer);
        for(int j=0;j<neuronInLayer;j++){
            neurons[i][j].id=createID(i,j);   
            neurons[i][j].value=0;
        }
    }

    connections=(connection*)malloc(sizeof(connection)*connectionNum);
    memcpy(connections,net.connections,sizeof(connection)*connectionNum);
    sort();
}

void network::serialize(std::string name){
  	FILE* f;
	f=fopen(name.c_str(),"wb+");
	fwrite(&layers,sizeof(int),1,f);
	fwrite(&neuronInLayer,sizeof(int),1,f);
	fwrite(&connectionNum,sizeof(int),1,f);
	fwrite(&id,sizeof(int),1,f);
	fwrite(connections,sizeof(connection),connectionNum,f);
	fclose(f);
}

void network::read(std::string name){
	FILE* f;
	f=fopen(name.c_str(),"rb");

	fread(&layers,sizeof(int),1,f);
	fread(&neuronInLayer,sizeof(int),1,f);
	fread(&connectionNum,sizeof(int),1,f);
	fread(&id,sizeof(int),1,f);
	fread(connections,sizeof(connection),15000,f);
	fclose(f);
    neurons=(neuron**)malloc(sizeof(neuron*)*layers);
    for(int i=0;i<layers;i++){
        if(i==0)neurons[i]=(neuron*)malloc(sizeof(neuron)*inputSize);
        else if(i==layers-1)neurons[i]=(neuron*)malloc(sizeof(neuron)*outputSize);
        else neurons[i]=(neuron*)malloc(sizeof(neuron)*neuronInLayer);
        for(int j=0;j<neuronInLayer;j++){
            neurons[i][j].id=createID(i,j);   
            neurons[i][j].value=0;
        }
    }
    connections=(connection*)malloc(sizeof(connection)*connectionNum);
    sort();
}
void network::destroy(){
    for(int i=0;i<layers;i++)
        free(neurons[i]);
    free(neurons);
    free(connections);
}
void network::add_connection(int amount){
    connectionNum+=amount;
    connections=(connection*)realloc(connections,sizeof(connection)*connectionNum);
    short src_id,src_layer,dest_id,dest_layer;
    for(int i=connectionNum;i<amount+connectionNum;i++){
        connections[i].weight=(rand()%200-100)/100.0;//-1.0 - 1.0 

        src_layer=rand()%(layers-1);//cant be last layer
        if(src_layer==0) src_id=rand()%inputSize;
        else src_id=rand()%neuronInLayer;

        dest_layer=rand()%(layers-1)+1;//cant be first layer
        if(dest_layer==layers-1) dest_id=rand()%outputSize;//0-9 
        else dest_id=rand()%neuronInLayer;

        connections[i].src_layer=src_layer;
        connections[i].src_id=src_id;
        connections[i].dest_layer=dest_layer;
        connections[i].dest_id=dest_id;
    }
    sort();
}
void network::rearrange_connection(int amount){
        //pick a connection position, reinitialize it with random values, sort all connections again
    int i; 
    short src_id,src_layer,dest_id,dest_layer;
    for(int _=0;_<amount;_++){
        i=rand()%connectionNum;
        connections[i].weight=(rand()%200-100)/100.0;//-1.0 - 1.0 
        src_layer=rand()%(layers-1);//cant be last layer
        if(src_layer==0) src_id=rand()%inputSize;
        else src_id=rand()%neuronInLayer;

        dest_layer=rand()%(layers-1)+1;//cant be first layer
        if(dest_layer==layers-1) dest_id=rand()%outputSize;//0-9 
        else dest_id=rand()%neuronInLayer;

        connections[i].src_layer=src_layer;
        connections[i].src_id=src_id;
        connections[i].dest_layer=dest_layer;
        connections[i].dest_id=dest_id;
    }
    sort();
}
