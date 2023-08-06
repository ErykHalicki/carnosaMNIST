#include <fstream>
#include <iostream>
#include "reading.h"
#include "net.h"
#include <cmath>
typedef unsigned char uchar;

void breakpoint(){
}
void print_image(uchar* image){
    char temp;
    for(int line=0;line<28;line++){
        for(int column=0;column<28;column++){
            if(image[line*28+column]>0)temp='1';
            else temp=' ';
            std::cout<<temp<<" ";
        }
        std::cout<<'\n';
    }
    std::cout<<'\n';
}

float evaluate(network net,uchar* labels, uchar** images,int setSize){
    float accuracy=0;

    float* result=(float*)malloc(sizeof(float)*outputSize);
    for(int n=0;n<setSize;n++){
        //print_image(images[n]);
        net.run(images[n],result);

        float best=0;
        uchar prediction;
        for(int i=0;i<outputSize;i++){
            //std::cout<<result[i];
            if(result[i]>best){
                best=result[i];
                prediction=i;
            }
        }
        if(prediction==labels[n])accuracy+=1.0;
    }
    free(result);
    return (accuracy/setSize)*100.0;//returns accuracy as a percentage
}
void evolve(int generations, int population,network* result){
    uchar** images=read_mnist_images(std::string("/Users/erykhalicki/desktop/projects/carnosa/mnist/mnist/t10k-images.idx3-ubyte"),1000,784);
    uchar* labels=read_mnist_labels(std::string("/Users/erykhalicki/desktop/projects/carnosa/mnist/mnist/t10k-labels.idx1-ubyte"),1000);

    network* networks=(network*)malloc(sizeof(network)*population);
    //all networks must have the same connection structure as the first rancomly generated network
    for(int i=0;i<population;i++){
        networks[i].init(4,300,15000);
    }

    float accuracies[population];
    float best_accuracy=0.0001;
    int last_increase=0;
    for(int gen=0;gen<generations;gen++){
        if(gen%50==0)
            std::cout<<"Generation "<<gen<<'\n';
        if(accuracies[0]>best_accuracy){
            std::cout<<"accuracy "<<(int)(accuracies[0]*100)/100.0<<"% +"<<(int)((accuracies[0]-best_accuracy)*100)/100.0<<"% "<<"after "<<gen-last_increase<<" Gen\n";
            best_accuracy=accuracies[0];
            last_increase=gen-1;
        }
        //srand (time(NULL));
        for(int n=0;n<population;n++){
            accuracies[n]=evaluate(networks[n],labels,images,100);
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
        //std::cout<<"Best Accuracy: "<<accuracies[0]<<"%"<<"\n";
        reproduce(networks[0],networks[1],&networks[population-4]);
        for(int i=2;i<population-4;i++){
            networks[i].rearrange_connection(100);
        }
        if(gen-last_increase>50){
            networks[0].rearrange_connection(100);
            networks[1].rearrange_connection(100);
        }
    }
    memcpy(result,networks,sizeof(network));
}


//
//add parent elimination to get out of ruts, find better ways to introduce
//competition
//time v0.1 vs v0.2 improvments


int main(int argc, char** argv){
    srand (time(NULL));
    std::cout<<"Carnosa_MNIST v0.2\n";
    network* best=(network*)malloc(sizeof(network));
    evolve(1000,8,best);
    best->serialize("serialTest.net");

    uchar** images=read_mnist_images(std::string("/Users/erykhalicki/desktop/projects/carnosa/mnist/mnist/t10k-images.idx3-ubyte"),10000,784);
    uchar* labels=read_mnist_labels(std::string("/Users/erykhalicki/desktop/projects/carnosa/mnist/mnist/t10k-labels.idx1-ubyte"),10000);

    std::cout<<evaluate(*best,labels,images,10000)<<"%";
    std::cout<<"done";
    int w;
    std::cin>>w;
   	return 0;	
}
/*
 *
 *Training: evolution based training 
 * train on first 5000, test on second 5000 
 * 
 * population of 10 competing, top 2 reproduce, bottom 4 get kicked out
 *
 * top 2 produce 4 offspring 
 * create 2 slightly randomized versions of themselves
 * and 2 with split genes, one child gets the first half of the connections
 * the other gets the second half, and vice versa
 *
 * randomization function, looks at all connnections
 * for about "Randomization %" of them
 * adds from -limit to +limit
 * defined as hyperparameter * called "Randomization Strength"
 *
 */
