#include "net.h"
#include <iostream>
#include <time.h>
#include <cmath>
#include <vector>
#include <unordered_set>
short create_id(int layer, int id){
    short res=0;
    res|=id;
    res<<=4;
    res|=layer;
    return res; 
}
void create_full_id(connection* c){
    c->full_id=0;
    c->full_id|=c->src;
    c->full_id<<=16;
    c->full_id|=c->dest;
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
void network::init(int l, int nil,int randomConnections,int k_s,int s){
    layers=l+3;
    //add max connection number (the amount of connections in a fully connected network)
    neuronInLayer=nil;
    neurons=(neuron**)malloc(sizeof(neuron*)*layers);
    connectionNum=randomConnections;
    if(input_width<k_s)k_s=input_width;
    kernel_size=k_s;
    stride=s;
    if(stride>kernel_size)stride=kernel_size;
    if(input_width%stride!=0)stride=1;
    convolution_layer_size=((input_width/stride)-k_s+1);
    kernel=(float**)malloc(sizeof(float*)*k_s);
    max_connections=convolution_layer_size*nil + pow(nil,l)+ nil*outputSize;
    connectionNum=std::min((long)connectionNum,max_connections);
    connection_ids=new std::unordered_set<int>;
    for(int i=0;i<k_s;i++){
        kernel[i]=(float*)malloc(sizeof(float)*k_s);
        for(int j=0;j<k_s;j++){
            kernel[i][j]=(rand()%200-100)/100.0;
        }
    }
    for(int i=0;i<layers;i++){
        if(i==0)neurons[i]=(neuron*)malloc(sizeof(neuron)*inputSize);//input layer
        else if(i==1)neurons[i]=(neuron*)malloc(sizeof(neuron)*pow(convolution_layer_size,2));//convolution layer
        else if(i==layers-1)neurons[i]=(neuron*)malloc(sizeof(neuron)*outputSize);// output layer
        else neurons[i]=(neuron*)malloc(sizeof(neuron)*neuronInLayer);//hidden layers
        for(int j=0;j<neuronInLayer;j++){
            neurons[i][j].id=create_id(i,j);
            neurons[i][j].value=0;
        }
    }
    
    connections=(connection*)malloc(sizeof(connection)*connectionNum);

    short src_id,src_layer,dest_id,dest_layer;
    for(int i=0;i<randomConnections;i++){
        connections[i].weight=(rand()%200-100)/100.0;//-1.0 - 1.0 

        src_layer=rand()%(layers-2)+1;//cant be last layer or input layer
        if(src_layer==0) src_id=rand()%inputSize;
        else src_id=rand()%neuronInLayer;

        dest_layer=std::max(src_layer+1,rand()%(layers-2)+2);//cant be input or convolution layer, must be after src_layer
        if(dest_layer==layers-1) dest_id=rand()%outputSize;//0-9 
        else dest_id=rand()%neuronInLayer;

        connections[i].src_layer=src_layer;
        connections[i].src_id=src_id;
        connections[i].dest_layer=dest_layer;
        connections[i].dest_id=dest_id;
        connections[i].dest=create_id(dest_layer,dest_id);
        connections[i].src=create_id(src_layer,src_id);
        create_full_id(connections+i);
        if(connection_ids->find(connections[i].full_id)==connection_ids->end())
            connection_ids->insert(connections[i].full_id);
        else i--;//redo the connection
    }
    //prevent duplicate connections?
    sort();
}
void network::clear(){
    int temp=inputSize;
    for(int i=0;i<layers;i++){
        if(i>1)temp=neuronInLayer;
        if(i==1)temp=pow(convolution_layer_size,2);
        if(i==layers-1)temp=outputSize;
        for(int j=0;j<temp;j++){
            neurons[i][j].value=0;
        }
    }
}
float network::inner_product(int offset_x, int offset_y){
    //+28 of any index is the next line
    float sum=0;
    for(int i=0;i<kernel_size;i++){
        for(int j=0;j<kernel_size;j++){
            sum+=neurons[0][(i+offset_y)*input_width+(j+offset_x)].value*kernel[i][j];
            //std::cout<<sum<<'\n';
        }
    }
    return sum;
}
void network::run(unsigned char* input, float* result){
    //loading input values to layer 0 
    clear();
    for(int i=0;i<inputSize;i++){
        neurons[0][i].value=std::min(255,(int)input[i])/255.0;
    }
    
    for(int i=0;i<convolution_layer_size;i++){
        for(int j=0;j<convolution_layer_size;j++){
            neurons[1][i*convolution_layer_size+j].value=inner_product(j*stride,i*stride);//setting convolution layer
        }
    }

    for(int i=0;i<connectionNum;i++){
        neurons[connections[i].dest_layer][connections[i].dest_id].value+=std::fmax(0,neurons[connections[i].src_layer][connections[i].src_id].value*connections[i].weight);
    }
    float sum=0;

    for(int i=0;i<outputSize;i++){
        sum+=neurons[layers-1][i].value;
    }
    for(int i=0;i<outputSize;i++){
        result[i]=(neurons[layers-1][i].value/sum)*100.0;
    }
}
void network::randomize(float multiplier){
    int rs= randomizationStrength*100;
   // srand (time(NULL));
    for(int i=0;i<connectionNum;i++){
        if((rand()%100+1)/100.0 <= randomizationRate*multiplier)
            connections[i].weight+=(rand()%((rs*2+1))-rs)/100.0;//from -rs to +rs
    }
}
void reproduce(network n1, network n2,network* offspring){
    int halfConnections=rand()%n1.connectionNum/2, allConnections=std::max(n1.connectionNum,n2.connectionNum);

    if(offspring[0].connectionNum<n2.connectionNum){
        offspring[0].connections=(connection*)realloc(offspring[0].connections,sizeof(connection)*allConnections);
        offspring[0].connectionNum=n2.connectionNum;
    }
    if(offspring[1].connectionNum<n1.connectionNum){
        offspring[1].connections=(connection*)realloc(offspring[1].connections,sizeof(connection)*allConnections);
        offspring[1].connectionNum=n1.connectionNum;
    }
    if(offspring[3].connectionNum<allConnections){
        offspring[3].connections=(connection*)realloc(offspring[3].connections,sizeof(connection)*allConnections);
        offspring[3].connectionNum=n2.connectionNum;
    }
    if(offspring[2].connectionNum<allConnections){
        offspring[2].connections=(connection*)realloc(offspring[2].connections,sizeof(connection)*allConnections);
        offspring[2].connectionNum=n1.connectionNum;
    }
    memcpy(offspring[0].connections,n1.connections,sizeof(connection)*n1.connectionNum);
    memcpy(offspring[1].connections,n2.connections,sizeof(connection)*n2.connectionNum);
    memcpy(&offspring[0].connections[halfConnections],&n2.connections[halfConnections],sizeof(connection)*(n2.connectionNum-halfConnections)); 
    memcpy(&offspring[1].connections[halfConnections],&n1.connections[halfConnections],sizeof(connection)*(n1.connectionNum-halfConnections)); 
    memcpy(offspring[2].connections,n1.connections,sizeof(connection)*n1.connectionNum);
    memcpy(offspring[3].connections,n2.connections,sizeof(connection)*n2.connectionNum);

    for(int i=0;i<offspring[0].kernel_size;i++){
        memcpy(offspring[0].kernel[i],n1.kernel[i],sizeof(float)*n1.kernel_size);
        memcpy(offspring[1].kernel[i],n2.kernel[i],sizeof(float)*n1.kernel_size);
        memcpy(offspring[2].kernel[i],n1.kernel[i],sizeof(float)*n1.kernel_size);
        memcpy(offspring[3].kernel[i],n2.kernel[i],sizeof(float)*n1.kernel_size);
    }
    for(int i=0;i<offspring[0].kernel_size/2;i++){
        for(int j=0;j<offspring[0].kernel_size/2;j++){
            offspring[0].kernel[i][j]=n2.kernel[i][j];
            offspring[1].kernel[i][j]=n1.kernel[i][j];
        }
    }
    offspring[2].randomize_kernel(0.1);
    offspring[3].randomize_kernel(0.1);

    for(int i=2;i<4;i++){
        offspring[i].rearrange_connection(100);
        offspring[i].randomize(1);
    }
}
void network::copy(network net){//TODO update copy function to copy new information
    layers=net.layers;
    neuronInLayer=net.neuronInLayer;
    neurons=(neuron**)malloc(sizeof(neuron*)*layers);
    connectionNum=net.connectionNum;
    for(int i=0;i<layers;i++){
        if(i==0)neurons[i]=(neuron*)malloc(sizeof(neuron)*inputSize);
        else if(i==layers-1)neurons[i]=(neuron*)malloc(sizeof(neuron)*outputSize);
        else neurons[i]=(neuron*)malloc(sizeof(neuron)*neuronInLayer);
        for(int j=0;j<neuronInLayer;j++){
            neurons[i][j].id=create_id(i,j);   
            neurons[i][j].value=0;
        }
    }

    connections=(connection*)malloc(sizeof(connection)*connectionNum);
    memcpy(connections,net.connections,sizeof(connection)*connectionNum);
    sort();
}

void network::serialize(std::string name){//TODO update serialization and reading function to work with convolution and the rest of new information
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

    connections=(connection*)malloc(sizeof(connection)*connectionNum);
	fread(&layers,sizeof(int),1,f);
	fread(&neuronInLayer,sizeof(int),1,f);
	fread(&connectionNum,sizeof(int),1,f);
	fread(&id,sizeof(int),1,f);
	fread(connections,sizeof(connection),connectionNum,f);
	fclose(f);
    neurons=(neuron**)malloc(sizeof(neuron*)*layers);
    for(int i=0;i<layers;i++){
        if(i==0)neurons[i]=(neuron*)malloc(sizeof(neuron)*inputSize);
        else if(i==layers-1)neurons[i]=(neuron*)malloc(sizeof(neuron)*outputSize);
        else neurons[i]=(neuron*)malloc(sizeof(neuron)*neuronInLayer);
        for(int j=0;j<neuronInLayer;j++){
            neurons[i][j].id=create_id(i,j);
            neurons[i][j].value=0;
        }
    }
    sort();
}
void network::destroy(){
    for(int i=0;i<layers;i++)
        free(neurons[i]);
    free(neurons);
    free(connections);
}
void network::add_connection(int amount){//TODO update and fix add connection function
    connectionNum+=amount;
    connections=(connection*)realloc(connections,sizeof(connection)*connectionNum);
    short src_id,src_layer,dest_id,dest_layer;
    int i;
    for(int j=0;j<amount;j++){
        i=j+connectionNum;
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
        //std::cout<<"added connection: "<<src_layer<<'\n';
        connections[i].dest=create_id(dest_layer,dest_id);
        connections[i].src=create_id(src_layer,src_id);
        
    }
    sort();
}
void network::rearrange_connection(int amount){//TODO update rearrange function
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
        connections[i].dest=create_id(dest_layer,dest_id);
        connections[i].src=create_id(src_layer,src_id);
        create_full_id(connections+i);
       if(connection_ids->find(connections[i].full_id)==connection_ids->end())
            connection_ids->insert(connections[i].full_id);
        else i--;//redo the connection
    }
    sort();
}

void network::reinforce(int result, float multiplier){//deprecated
    std::vector<int> rewarded;
    int rewarded_num=0;
    std::unordered_set<short> connected,done;
    for(int i=0;i<connectionNum;i++){
        if(connections[i].dest_layer==layers-1 && connections[i].dest_id==result){
            rewarded.push_back(i);
            rewarded_num++;
            connected.insert(connections[i].src);
        }
    }
    bool added=true;
    int tmp;
    while(added){
        added=false;
        for(int l=0;l<layers-1;l++){
            if(l==0)tmp=inputSize;
            else tmp=neuronInLayer;
            for(int i=0;i<tmp;i++){
                if(connected.find(neurons[l][i].id)!=connected.end()&&l!=0 && done.find(neurons[l][i].id)==done.end()){
                    for(int c=0;c<connectionNum;c++){
                        if(connections[c].dest==neurons[l][i].id){
                            if(done.find(connections[c].src)==done.end())
                                connected.insert(connections[c].src);
                            rewarded.push_back(c);
                            rewarded_num++;
                        }
                    }
                    added=true;
                    done.insert(neurons[l][i].id);
                }
            }
        }
    }
    //std::cout<<rewarded_num<<" connections rewarded\n";
    for(int i=0;i<rewarded_num;i++){
        connections[rewarded[i]].weight*=multiplier;
    }
}

//TODO add kernel evolution functions
//TODO run a profiler
void network::randomize_kernel(float percentage){
    int rs= randomizationStrength*100;

    for(int i=0;i<kernel_size;i++){
        for(int j=0;j<kernel_size;j++){
            if((rand()%100+1)/100.0 <= percentage){
                kernel[i][j]+=(rand()%((rs*2+1))-rs)/100.0;//from -rs to +rs
            }
        }
    }        
}
