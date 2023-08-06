#include <fstream>
#include <iostream>
#include "reading.h"
#include "net.h"
#include <cmath>
#include <vector>
#include <thread>
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
        temp=std::min(9,(int)net.neurons[1][i].value);
        if(temp==0)std::cout<<"_ ";
        else std::cout<<std::min(9,(int)net.neurons[1][i].value)<<" ";
    }
}

void evaluate(bool use_score,network net,uchar* labels, uchar** images,int setSize,int offset,float* output){
    float score=0;

    float* result=(float*)malloc(sizeof(float)*outputSize);
    for(int n=offset;n<offset+setSize;n++){
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
        float avg=0;
        for(int i=0;i<outputSize;i++){
            if(i!=prediction)
            avg+=result[i]/(outputSize-1);
        }
        if(prediction==labels[n]){
            //std::cout<<"best: "<<best<<" avg: "<<avg<<'\n';
            if(use_score){score+=best;}//more confident scores result in significantly more points
            else{score++;}
        }
        else if(use_score){
            score-=best;
        }
    }
    free(result);
    //add weighted (least squares style?) accuracy
    //the more confidently the model picks a correct answer, the better the score
    // example: 
    //
    //      correct answer: 1
    //      49% - 0
    //      51% - 1
    //      rewarded very little
    //
    //      correct answer: 0
    //      85% - 0
    //      15% - 1
    //      rewarded a lot
    //      
    //      promotes strong development answers
    //
    //      also works in inverse, if the result is predicted poorly with strong confidence, 
    //      score can be deducted
    //    
    //      allows a greater level of precision in fitness testing
    //      now running only 10 random images can give a fitness more accurate than just 10% increments
    //
    if(use_score) *output=score;
    else *output=(score/setSize)*100.0;//returns score as accuracy percentage
}
void evolve(network seed,int generations, int population,network* result){
    uchar** images=read_mnist_images(std::string("/Users/erykhalicki/desktop/projects/carnosa/mnist/mnist/t10k-images.idx3-ubyte"),10000,784);
    uchar* labels=read_mnist_labels(std::string("/Users/erykhalicki/desktop/projects/carnosa/mnist/mnist/t10k-labels.idx1-ubyte"),10000);

    network* networks=(network*)malloc(sizeof(network)*population);
    //all networks must have the same connection structure as the first rancomly generated network
    for(int i=0;i<population;i++){
//      networks[i].copy(seed);
        networks[i].init(2,150,10000,3,1);
    }

    float accuracies[population];
    float best_accuracy=0;
    int last_increase=0;
    int offset=0;
    std::thread threads[population];
    for(int gen=0;gen<generations;gen++){
        if(gen%50==0){
            std::cout<<"Generation "<<gen<<'\n';
            srand (time(NULL));
            offset=rand()%9900;
        }
                //srand (time(NULL));
        for(int n=0;n<population;n++){
            //TODO ADD THREADING RIGHT HERE
            //accuracies[n]=evaluate(true,networks[n],labels,images,100,0);
            threads[n]=std::thread(evaluate,false,networks[n],labels,images,1000,0,&accuracies[n]);
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
        //std::cout<<"Best score: "<<accuracies[0]<<"%"<<"\n";
        reproduce(networks[0],networks[1],&networks[population-4]);
        //reproduce(networks[0],networks[2],&networks[population-8]);
        if(accuracies[0]>best_accuracy||gen==0){
            std::cout<<"new high score "<<(int)(accuracies[0]*100)/100.0<<" +"<<(int)((accuracies[0]-best_accuracy)*100)/100.0<<"% "<<"after "<<gen-last_increase<<" Gen\n";
            best_accuracy=accuracies[0];
            last_increase=gen-1;
        }

        if(gen-last_increase>50){
            for(int i=1;i<population;i++)
                networks[i].rearrange_connection(100);
                last_increase=gen; 
                //networks[0].add_connection(50);
                //networks[1].add_connection(50);
        }
        else{
            for(int i=2;i<population-4;i++){
                networks[i].rearrange_connection(100);
            }
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
    std::cout<<"Carnosa_MNIST v0.3\n";
    network* best=(network*)malloc(sizeof(network));
    network seed;
    seed.init(2,200,15000,4,2);

    //seed.read("serialTest.net");
    evolve(seed,1000,8,best);
    //
    //best->serialize("serialTest.net");

    uchar** images=read_mnist_images(std::string("/Users/erykhalicki/desktop/projects/carnosa/mnist/mnist/t10k-images.idx3-ubyte"),10000,784);
    uchar* labels=read_mnist_labels(std::string("/Users/erykhalicki/desktop/projects/carnosa/mnist/mnist/t10k-labels.idx1-ubyte"),10000);
        //best->read("serialTest.net");
    float res;
    evaluate(false, *best,labels,images,10000,0,&res);
    std::cout<<res<<"%";
    std::cout<<"\ndone \n";
   	return 0;	
}
/*
 *
 *Training: evolution based training 
 * train on first 5000, test on second 5000 
 * 
 * population competing, top few reproduce, bottom few get kicked out
 *
 * 2 produce 4 offspring 
 * create 2 slightly randomized versions of themselves
 * and 2 with split genes, one child gets the first part of the connections
 * the other gets the second part, and vice versa, with a random switching point
 *
 * randomization function, looks at all connnections
 * for about "Randomization %" of them
 * adds from -limit to +limit
 * defined as hyperparameter * called "Randomization Strength"
 *
 */


/* Version list
 * v0.1 - constant connection structure, constant connection size, 
 *      ~30% accuracy on training data
 *
 * v0.2 - mutable connection Structure, constant connection size 
 *      ~50% accuracy on training data
 *
 * v0.3 - mutable connection structure, constant connection size
 *      - Added convolution
 *      - added weighted accuracy (more confident answers have higher weight)
 *      ~50% accuracy on training data
 *
 * v0.4
 *
 *
 */
