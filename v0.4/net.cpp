#include "net.h"
#include <iostream>
#include <time.h>
#include <cmath>
#include <Accelerate/Accelerate.h>

void network::init(int l, int ninl,int k_s,int s){
    layers=l+4;
    connectionNum=pow(convolution_layer_size,2)*ninl + pow(ninl,l)+ ninl*outputSize;
    neuronInLayer=ninl;
    kernel_size=k_s;
    stride=s;
    convolution_layer_size=((input_width/stride)-k_s+1);
    pooling_layer_size=((convolution_layer_size/2)-1);

    neurons=(float**)malloc(sizeof(float*)*layers);
    layer_size=(int*)malloc(sizeof(int*)*layers);
    kernel=(float**)malloc(sizeof(float*)*k_s);

    for(int i=0;i<k_s;i++){
        kernel[i]=(float*)malloc(sizeof(float)*k_s);
        for(int j=0;j<k_s;j++){
            kernel[i][j]=1;
        }
    }
    weights=(float**)malloc(sizeof(float*)*(layers-3));
    int conn_num=0;
    for(int i=0;i<layers;i++){
        if(i==0){
            neurons[i]=(float*)malloc(sizeof(float)*inputSize);//input layer
            layer_size[i]=inputSize;
        }
        else if(i==1){
            neurons[i]=(float*)malloc(sizeof(float)*(pow(convolution_layer_size,2)+1));//convolution layer
            layer_size[i]=pow(convolution_layer_size,2);
        }
        else if(i==2){
            neurons[i]=(float*)malloc(sizeof(float)*(pow(pooling_layer_size,2)+1));//pooling layer
            layer_size[i]=pow(pooling_layer_size,2);
        }
        else if(i==layers-1){
            neurons[i]=(float*)malloc(sizeof(float)*outputSize);// output layer
            layer_size[i]=outputSize;
        }
        else {
            neurons[i]=(float*)malloc(sizeof(float)*neuronInLayer+1);//hidden layers
            layer_size[i]=neuronInLayer;
        }
    }
    for(int i=0;i<layers-3;i++){
        weights[i]=(float*)malloc(sizeof(float)*layer_size[i+3]*(layer_size[i+2]+1));
        for(int j=0;j<layer_size[i+3]*(layer_size[i+2]+1);j++){
            weights[i][j]=1;
        }
    }
    //std::cout<<"connectionNum: "<<connectionNum<<'\n';
    clear();
}
void network::clear(){
    for(int i=0;i<layers;i++){
        for(int j=0;j<layer_size[i];j++){
            neurons[i][j]=0;
        }
    }
}
float network::inner_product(int offset_x, int offset_y){
    float sum=0;
    for(int i=0;i<kernel_size;i++){
        for(int j=0;j<kernel_size;j++){
            sum+=neurons[0][(i+offset_y)*input_width+(j+offset_x)]*kernel[i][j];
        }
    }
    return sum;
}
float network::max_pool(int offset_x, int offset_y){
    float max=0;
    for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            max=fmax(max,neurons[1][(i+offset_y)*convolution_layer_size+(j+offset_x)]);
        }
    }
    return max;
}

inline float activation(float inp){
    return std::fmax(0,inp);
    //return 1/(1+pow(2.718,-inp*3));
}

void network::run(unsigned char* input, float* result){
    clear();
    for(int i=0;i<inputSize;i++){
        neurons[0][i]=std::min(255,(int)input[i])/255.0;//stored row major
    }
    
    for(int i=0;i<convolution_layer_size;i++){
        for(int j=0;j<convolution_layer_size;j++){
            neurons[1][i+j*convolution_layer_size]=inner_product(i*stride,j*stride);//setting convolution layer
        }
    }
    for(int i=0;i<pooling_layer_size;i++){
        for(int j=0;j<pooling_layer_size;j++){
            neurons[2][i+j*pooling_layer_size]=max_pool(i*2,j*2);//setting pooling layer
        }
    }

    for(int i=0;i<layers-3;i++){
        vDSP_mmul(weights[i],1,neurons[i+2],1,neurons[i+3],1,layer_size[i+3],1,layer_size[i+2]+1);
        for(int j=0;j<layer_size[i+3];j++){
            neurons[i+3][j]=activation(neurons[i+3][j]);//ReLu
        }
    }
    float sum=0;
    for(int i=0;i<outputSize;i++){
        sum+=neurons[layers-1][i];
    }
    for(int i=0;i<outputSize;i++){
        result[i]=fmax(0,(neurons[layers-1][i]/sum)*100.0);
    }
}

void network::randomize(float multiplier){
    int rs= randomizationPower*100;
    for(int l=0;l<layers-3;l++){
        //if((rand()%100)/100.0<randomizationRate*multiplier)
            //neurons[l+1][layer_size[l+1]]+=(rand()%((rs*2+1))-rs)/100.0;;
        for(int i=0;i<(layer_size[l+2]+1)*layer_size[l+3]*std::fmax(1,randomizationRate*multiplier);i++){
            int pos=rand()%(layer_size[l+2]+1)*layer_size[l+3];
            //if((pos+1)%(layer_size[l+1]+1)!=0)
                weights[l][pos]+=(rand()%((rs*2+1))-rs)/100.0;//from -rs to +rs
            //last layer_size[l+1] weights must be left alone, they are bias values 
            //this means the matrices will be stored row major
            //example: 
            // | 1 2 | --> | 1 2 3 4 |
            // | 3 4 |
        }
    }
    randomize_kernel(0.1*multiplier);
}

void reproduce(network n1, network n2,network* offspring){
    for(int i=0;i<n1.layers-3;i++){
        memcpy(offspring[0].weights[i],n1.weights[i],sizeof(float)*n1.layer_size[i+2]*(n1.layer_size[i+1]+1)/2);
        memcpy(offspring[0].weights[i],&n2.weights[i][n1.layer_size[i+2]*(n1.layer_size[i+1]+1)/2 - 1],sizeof(float)*n1.layer_size[i+2]*(n1.layer_size[i+1]+1)/2);
        memcpy(offspring[1].weights[i],n2.weights[i],sizeof(float)*n1.layer_size[i+2]*(n1.layer_size[i+1]+1)/2);
        memcpy(offspring[1].weights[i],&n1.weights[i][n1.layer_size[i+2]*(n1.layer_size[i+1]+1)/2 - 1],sizeof(float)*n1.layer_size[i+2]*(n1.layer_size[i+1]+1)/2);
        memcpy(offspring[2].weights[i],n1.weights[i],sizeof(float)*n1.layer_size[i+2]*(n1.layer_size[i+1]+1));
        memcpy(offspring[3].weights[i],n2.weights[i],sizeof(float)*n1.layer_size[i+2]*(n1.layer_size[i+1]+1));
        memcpy(offspring[4].weights[i],n1.weights[i],sizeof(float)*n1.layer_size[i+2]*(n1.layer_size[i+1]+1));
        for(int j=0;j<(n1.layer_size[i+2]+1)*n1.layer_size[i+3];j++){
            offspring[4].weights[i][j]+=n2.weights[i][j];
            offspring[4].weights[i][j]/=2;
        }
        /*offspring[0].neurons[i+1][offspring[0].layer_size[i+1]]=n1.neurons[i+1][n1.layer_size[i+1]];
        offspring[1].neurons[i+1][offspring[1].layer_size[i+1]]=n2.neurons[i+1][n1.layer_size[i+1]];
        offspring[2].neurons[i+1][offspring[2].layer_size[i+1]]=n1.neurons[i+1][n1.layer_size[i+1]];
        offspring[3].neurons[i+1][offspring[3].layer_size[i+1]]=n2.neurons[i+1][n1.layer_size[i+1]];
        */
    }

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

    for(int i=2;i<4;i++){
        offspring[i].randomize(1);
    }
}

void network::copy(network net){    
    layers=net.layers;
    neuronInLayer=net.neuronInLayer;
    connectionNum=net.connectionNum;
    kernel_size=net.kernel_size;
    stride=net.stride;
    convolution_layer_size=net.convolution_layer_size;
    pooling_layer_size=net.pooling_layer_size;

    weights=(float**)malloc(sizeof(float*)*(layers-3));
    neurons=(float**)malloc(sizeof(float*)*layers);
    layer_size=(int*)malloc(sizeof(int*)*layers);
    kernel=(float**)malloc(sizeof(float*)*kernel_size);

    for(int i=0;i<kernel_size;i++){
        kernel[i]=(float*)malloc(sizeof(float)*kernel_size);
        for(int j=0;j<kernel_size;j++){
            kernel[i][j]=net.kernel[i][j];
        }
    }
    for(int i=0;i<layers;i++){
        if(i==0){
            neurons[i]=(float*)malloc(sizeof(float)*inputSize);//input layer
            layer_size[i]=inputSize;
        }
        else if(i==1){
            neurons[i]=(float*)malloc(sizeof(float)*pow(convolution_layer_size,2)+1);//convolution layer
            layer_size[i]=pow(convolution_layer_size,2);
        }
        else if(i==2){
            neurons[i]=(float*)malloc(sizeof(float)*(pow(pooling_layer_size,2)+1));//pooling layer
            layer_size[i]=pow(pooling_layer_size,2);
        }

        else if(i==layers-1){
            neurons[i]=(float*)malloc(sizeof(float)*outputSize);// output layer
            layer_size[i]=outputSize;
        }
        else {
            neurons[i]=(float*)malloc(sizeof(float)*neuronInLayer+1);//hidden layers
            layer_size[i]=neuronInLayer;
        }
    }
    for(int i=0;i<layers-3;i++){
        weights[i]=(float*)malloc(sizeof(float)*layer_size[i+2]*(layer_size[i+1]+1));
        for(int j=0;j<layer_size[i+2]*(layer_size[i+1]+1);j++){
            weights[i][j]=net.weights[i][j];
        }
    }
    clear();
}

void network::serialize(std::string name){//TODO update serialization and reading function to work with convolution and the rest of new information
    /*
  	FILE* f;
	f=fopen(name.c_str(),"wb+");
	fwrite(&layers,sizeof(int),1,f);
	fwrite(&neuronInLayer,sizeof(int),1,f);
	fwrite(&connectionNum,sizeof(int),1,f);
	fwrite(&id,sizeof(int),1,f);
	fwrite(connections,sizeof(connection),connectionNum,f);
	fclose(f);
    */
}

void network::read(std::string name){
    /*
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
    */
}

void network::randomize_kernel(float percentage){
    int rs= randomizationPower*100;
    for(int i=0;i<kernel_size;i++){
        for(int j=0;j<kernel_size;j++){
            if((rand()%100+1)/100.0 <= percentage){
                kernel[i][j]+=(rand()%((rs*2+1))-rs)/100.0;//from -rs to +rs
            }
        }
    }        
}
