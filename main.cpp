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

void evaluate(network net,uchar* labels, uchar** images,int setSize,int offset,float* output, bool training){
    float score=0,best=0, *result=(float*)malloc(sizeof(float)*outputSize);
    unsigned long prediction=0;
    int correct;
    for(int n=offset;n<offset+setSize;n++){
        net.run(images[n],result);
        vDSP_maxvi(result,1,&best,&prediction,outputSize);
        if(training){
            for(int i=0;i<outputSize;i++){
                correct=0;
                if(i==labels[n])correct=100;
                score-=pow(correct-result[i],2)/2;
            }
        }
        else if(labels[n]==prediction)score++;
    }
    free(result);
    if(!training)
    *output=(score/setSize)*100.0;//returns score as accuracy percentage
    else 
        *output=(score/(setSize*setSize));//returns score as accuracy percentage
}

void evolve(network seed,int generations, int population,network* result){
    uchar** images=read_mnist_images(std::string("/Users/erykhalicki/desktop/projects/carnosa/mnist/data/train-images.idx3-ubyte"),60000,784);
    uchar* labels=read_mnist_labels(std::string("/Users/erykhalicki/desktop/projects/carnosa/mnist/data/train-labels.idx1-ubyte"),60000);
    uchar** test_images=read_mnist_images(std::string("/Users/erykhalicki/desktop/projects/carnosa/mnist/data/t10k-images.idx3-ubyte"),60000,784);
    uchar* test_labels=read_mnist_labels(std::string("/Users/erykhalicki/desktop/projects/carnosa/mnist/data/t10k-labels.idx1-ubyte"),60000);

    network* networks=(network*)malloc(sizeof(network)*population);
    for(int i=0;i<population;i++){
        networks[i].copy(seed);
    }
    float accuracies[population];
    float best_accuracy=0;
    float best_accuracy_test=0;
    int last_increase=0;
    int last_increase_test=0;
    int offset=0;
    int size=100;
    std::thread threads[population];
    int gen=0;
    while(gen<generations || gen-last_increase_test<5000){
    //for(int gen=0;gen<generations;gen++){
        if(gen%50==0 && gen>0){
            //srand (time(NULL));
            offset=rand()%(60000-size);
            float res;
            evaluate(networks[0],test_labels,test_images,10000,0,&res,false);
            if(res>best_accuracy_test){
                last_increase_test=gen;
                best_accuracy_test=res;
            }
            std::cout<<"Generation "<<gen<<" best accuracy: "<<res<<'\n';
        }
        for(int n=0;n<population;n++){
            threads[n]=std::thread(evaluate,networks[n],labels,images,size,offset,&accuracies[n],true);
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
        }

        for (int n = 0; n < population; n++) {
                //std::cout<<accuracies[n]<<"%"<<"\n";
        }
        //std::cout<<"Best score: "<<accuracies[0]<<"%"<<" gen "<<gen<<"\n";
        for(int i=0;i<population/6+1;i++){
            reproduce(networks[i*2],networks[i*2+1],&networks[population-(i+1)*4]);
        }
        if(accuracies[0]>best_accuracy||gen==0){
            std::cout<<"new high score "<<(int)(accuracies[0]*10000)/10000.0<<" +"<<(int)((accuracies[0]-best_accuracy)*10000)/10000.0<<"% "<<"after "<<gen-last_increase<<" Gen\n";
            best_accuracy=accuracies[0];
            last_increase=gen;
        }
        for(int i=2;i<population;i++)
            networks[i].randomize((gen-last_increase)/50.0);

        gen++;
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
    /*
    float test[outputSize];
    seed.run(images[0],test);
    for(int i=0;i<outputSize;i++)
    std::cout<<test[i]<<'\n';
    */
    evolve(seed,1000,20,best);
    float res;
    evaluate(seed,labels,images,10000,0,&res,false);
    std::cout<<"Original Network Accuracy: "<<res<<"%\n";
    evaluate(*best,labels,images,10000,0,&res,false);
    std::cout<<"Final Network Accuracy: "<<res<<"%";
   	return 0;	
}
