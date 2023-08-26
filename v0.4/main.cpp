#include <fstream>
#include <iostream>
#include "reading.h"
#include "net.h"
#include <cmath>
#include <vector>
#include <thread>
#include <Accelerate/Accelerate.h>

typedef unsigned char uchar;

void print_image(uchar* image){
    char temp;
    for(int line=0;line<28;line++){
        for(int column=0;column<28;column++){
            if(image[line*28+column]>0)temp='1';
            else temp='_';
            std::cout<<temp<<" ";
        }
        std::cout<<'\n';
    }
    std::cout<<'\n';
}

void print_convolution(network net, uchar* image){
    float* result=(float*)malloc(sizeof(float)*outputSize);
    net.run(image,result);
    char temp;
    for(int i=0;i<pow(net.convolution_layer_size,2);i++){
        if(i%net.convolution_layer_size==0){std::cout<<'\n';}
        temp=std::min(9,(int)net.neurons[1][i]);
        if(temp==0)std::cout<<"_ ";
        else std::cout<<std::min(9,(int)net.neurons[1][i])<<" ";
    }
}

void evaluate(network net,uchar* labels, uchar** images,int setSize,int offset,float* output){
    float score=0,best=0, *result=(float*)malloc(sizeof(float)*outputSize);
    unsigned long prediction=0;
    for(int n=offset;n<offset+setSize;n++){
        net.run(images[n],result);
        vDSP_maxvi(result,1,&best,&prediction,outputSize);
        if(labels[n]==prediction) score++;
    }
    free(result);
    *output=(score/setSize)*100.0;//returns score as accuracy percentage
}

void evolve(network seed,int generations, int population,network* result){
    uchar** images=read_mnist_images(std::string("/Users/erykhalicki/desktop/projects/carnosa/mnist/data/train-images.idx3-ubyte"),60000,784);
    uchar* labels=read_mnist_labels(std::string("/Users/erykhalicki/desktop/projects/carnosa/mnist/data/train-labels.idx1-ubyte"),60000);

    network* networks=(network*)malloc(sizeof(network)*population);
    for(int i=0;i<population;i++){
        networks[i].copy(seed);
    }
    float accuracies[population];
    float best_accuracy=0;
    int last_increase=0;
    int offset=0;
    std::thread threads[population];
    for(int gen=0;gen<generations;gen++){
        if(gen%50==0 && gen>0){
            std::cout<<"Generation "<<gen<<" best accuracy: "<<accuracies[0]<<'\n';
            srand (time(NULL));
            offset+=5;
        }
        for(int n=0;n<population;n++){
            threads[n]=std::thread(evaluate,networks[n],labels,images,500,0,&accuracies[n]);
        }  
        for(int n=0;n<population;n++){
            threads[n].join();
        }
        for (int i = 0; i < population-1; i++) {
            for (int j = 0; j < population-i-1; j++) {
                if (accuracies[j] < accuracies[j + 1]) {
                    float tempf=accuracies[j];
                    network tempn=networks[j];
                    networks[j]=networks[j+1];
                    networks[j+1]=tempn;
                    accuracies[j]=accuracies[j+1];
                    accuracies[j+1]=tempf;
                }
            
        }

        for (int n = 0; n < population; n++) {
                //std::cout<<accuracies[n]<<"%"<<"\n";
        }
        //std::cout<<"Best score: "<<accuracies[0]<<"%"<<"\n";
        reproduce(networks[0],networks[1],&networks[population-4]);
        if(accuracies[0]>best_accuracy||gen==0){
            std::cout<<"new high score "<<(int)(accuracies[0]*100)/100.0<<" +"<<(int)((accuracies[0]-best_accuracy)*100)/100.0<<"% "<<"after "<<gen-last_increase<<" Gen\n";
            best_accuracy=accuracies[0];
            last_increase=gen;
        }
        for(int i=1;i<population;i++)
            networks[i].randomize((gen-last_increase)/50.0);
    }
    memcpy(result,networks,sizeof(network));
}

int main(int argc, char** argv){
    srand (time(NULL));
    std::cout<<"Carnosa_MNIST v0.4\n";
    uchar** images=read_mnist_images(std::string("/Users/erykhalicki/desktop/projects/carnosa/mnist/data/t10k-images.idx3-ubyte"),10000,784);
    uchar* labels=read_mnist_labels(std::string("/Users/erykhalicki/desktop/projects/carnosa/mnist/data/t10k-labels.idx1-ubyte"),10000);

    network* best=(network*)malloc(sizeof(network));
    network seed;
    seed.init(1,100,4,1);
    seed.randomize(100);
    /*
    float test[outputSize];
    seed.run(images[0],test);
    for(int i=0;i<outputSize;i++)
    std::cout<<test[i]<<'\n';
    */
    evolve(seed,3000,10,best);
    float res;
    evaluate(seed,labels,images,10000,0,&res);
    std::cout<<"Original Network Accuracy: "<<res<<"%\n";
    evaluate(*best,labels,images,10000,0,&res);
    std::cout<<"Final Network Accuracy: "<<res<<"%";
   	return 0;	
}
