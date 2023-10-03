#include <fstream>
#include <iostream>
#include "reading.h"
#include "writing.h"
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
/*
void print_pool(network net, uchar* image){
    float* result=(float*)malloc(sizeof(float)*outputSize);
    net.run(image,result);
    char temp;
    for(int i=0;i<pow(net.pooling_layer_size,2);i++){
        if(i%net.pooling_layer_size==0){std::cout<<'\n';}
        temp=std::min(9,(int)net.neurons[2][i]);
        if(temp==0)std::cout<<"_ ";
        else std::cout<<std::min(9,(int)net.neurons[2][i])<<" ";
    }
}
*/
void evaluate(network net,uchar* labels, uchar** images,int setSize,int offset,float* output, bool training){
    float score=0,best=0, *result=(float*)malloc(sizeof(float)*outputSize);
    unsigned long prediction=0;
    int correct;
    for(int n=offset;n<offset+setSize;n++){
        net.run(images[n],result);
        vDSP_maxvi(result,1,&best,&prediction,outputSize-1);
        if(training){
            for(int i=0;i<outputSize-1;i++){
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
int run_population(network** networks,int islands, int population,uchar* image){
    int best_pick=0;
    float max=0,*result=(float*)malloc(sizeof(float)*outputSize);
    for(int n=0;n<islands;n++){
        for(int i=0;i<population;i++){
            networks[n][i].run(image, result);
            for(int j=0;j<outputSize;j++){
                if(result[j]>max){
                    max=result[j];
                    best_pick=j;
                }
            }
        }
    }
    return best_pick;
}

void evaluate_population(network** nets,uchar* labels, uchar** images,int setSize,int offset,float* output, int population, int islands){
    float score=0; 
    for(int n=offset;n<offset+setSize;n++){
        if(labels[n]==run_population(nets,islands,population,images[n]))score++;
    }
    *output=(score/setSize)*100.0;//returns score as accuracy percentage
}
void evolve(network seed,int generations, int population,network* result){
    uchar** images=read_mnist_images(std::string("/Users/erykhalicki/desktop/projects/current/carnosa/mnist/data/train-images.idx3-ubyte"),60000,784);
    uchar* labels=read_mnist_labels(std::string("/Users/erykhalicki/desktop/projects/current/carnosa/mnist/data/train-labels.idx1-ubyte"),60000);
    uchar** test_images=read_mnist_images(std::string("/Users/erykhalicki/desktop/projects/current/carnosa/mnist/data/t10k-images.idx3-ubyte"),10000,784);
    uchar* test_labels=read_mnist_labels(std::string("/Users/erykhalicki/desktop/projects/current/carnosa/mnist/data/t10k-labels.idx1-ubyte"),10000);

    
    int island_num=1;
    float accuracies[island_num][population];
    float best_accuracy[island_num];
    float best_accuracy_test[island_num];
    int last_increase[island_num];
    int last_increase_test[island_num];
    int offset=0;
    int size=100;
    std::thread threads[population];
    int gen=0;
    std::string data="";
    data.reserve(100);
    start_data("data.csv");
    float res;
    network top[island_num];
    network** networks=(network**)malloc(sizeof(network*)*island_num);
    for(int i=0;i<island_num;i++){
        networks[i]=(network*)malloc(sizeof(network)*population);
    }
    for(int l=0;l<island_num;l++){
        for(int i=0;i<population;i++){
            networks[l][i].copy(seed);
        }   
    }
    while(gen<generations || gen-last_increase_test[0]<2000){
        if(gen%50==0) std::cout<<"Generation "<<gen<<"\n";
        for(int island=0;island<island_num;island++){
        if(gen%50==0){
            //srand (time(NULL));
            //offset+=10;
            evaluate(networks[island][0],labels,images,10000,49999,&res,false);
            //if(randomizationPower<3)randomizationPower+=0.02;
            if(res>best_accuracy_test[island]){
                last_increase_test[island]=gen;
                top[island].copy(networks[island][0]);
                best_accuracy_test[island]=res;
                //size+=5;
            }
            else offset=rand()%(50000-size);
            //for(int i=0;i<population;i++)networks[i].copy(top);
            std::cout<<"\tIsland "<<island<<" best accuracy: "<<res<<"% vs top "<<best_accuracy_test[island]<<"% \n";
        }
        for(int n=0;n<population;n++){
            threads[n]=std::thread(evaluate,networks[island][n],labels,images,size,offset,&accuracies[island][n],true);
        }  
        for(int n=0;n<population;n++){
            threads[n].join();
        }
        //if(accuracies[0]<best_accuracy && gen>5)accuracies[0]=best_accuracy;//eltist selection, if the best network performs worse in this generation, it cannot be replaced
        for (int i = 0; i < population-1; i++) {
            for (int j = 0; j < population-i-1; j++) {
                if (accuracies[island][j] < accuracies[island][j + 1]) {
                    float tempf=accuracies[island][j];
                    network tempn=networks[island][j];
                    networks[island][j]=networks[island][j+1];
                    networks[island][j+1]=tempn;
                    accuracies[island][j]=accuracies[island][j+1];
                    accuracies[island][j+1]=tempf;
                }      
            }
        }

        /*
        for (int n = 0; n < population; n++) {
                //std::cout<<accuracies[n]<<"%"<<"\n";
        }
        for(int i=0;i<population/7+1;i++){
            reproduce(networks[i*2],networks[i*2+1],&networks[population-(i+1)*5]);
        }

        */

        //std::cout<<"Best score: "<<accuracies[island][0]<<"%"<<" gen "<<gen<<"\n";
                //reproduce(networks[0],networks[population-1],&networks[population-20]);
        if(gen>0 && (gen%80==0 ||gen%81==0||gen%82==0))
        reproduce(top[rand()%island_num],networks[island][0],&networks[island][population-5]);
        else
        reproduce(top[island],networks[island][0],&networks[island][population-5]);
        reproduce(networks[island][0],networks[island][1],&networks[island][population-10]);
        reproduce(networks[island][0],networks[island][2],&networks[island][population-15]);
        reproduce(networks[island][0],networks[island][3],&networks[island][population-20]);
        reproduce(networks[island][0],networks[island][4],&networks[island][population-25]);
        reproduce(networks[island][0],networks[island][5],&networks[island][population-30]);
        reproduce(networks[island][0],networks[island][1],&networks[island][population-35]);
        //reproduce(networks[island][0],networks[island][2],&networks[island][population-15]);
        if(accuracies[island][0]>best_accuracy[island]||gen==0){
            //std::cout<<"new high score "<<(int)(accuracies[island][0]*10000)/10000.0<<" +"<<(int)((accuracies[island][0]-best_accuracy[island])*10000)/10000.0<<"% "<<"after "<<gen-last_increase[island]<<" Gen\n";
            best_accuracy[island]=accuracies[island][0];
            last_increase[island]=gen;
        }
        for(int i=2;i<population;i++)
            networks[island][i].randomize(0.25+(gen-last_increase[island])/50.0);

        //end of generation code
        add_data(&data,island);
        add_data(&data,best_accuracy[island]);
        add_data(&data,best_accuracy_test[island]);
        add_data(&data,res);
        add_data(&data,accuracies[island][0]);
        add_data(&data,accuracies[island][1]);
        add_data(&data,randomizationPower);
        write_data(data);
        data="";
    }
        if(gen%200==0){
            evaluate_population(networks,labels,images,100,49999,&res,population,island_num);
            std::cout<<"Generation "<<gen<<" population accuracy: "<<res<<"% \n";
        }
        gen++;
    }
    end_data();
    memcpy(result,networks[0],sizeof(network));
}

int main(int argc, char** argv){
    srand (time(NULL));
    std::cout<<"Carnosa_MNIST v0.4\n";
    uchar** images=read_mnist_images(std::string("/Users/erykhalicki/desktop/projects/current/carnosa/mnist/data/t10k-images.idx3-ubyte"),10000,784);
    uchar* labels=read_mnist_labels(std::string("/Users/erykhalicki/desktop/projects/current/carnosa/mnist/data/t10k-labels.idx1-ubyte"),10000);

    network* best=(network*)malloc(sizeof(network));
    network seed;
    seed.init(1,200,4,1);
    seed.randomize(10);
    float test[outputSize];
    int choice=rand()%1000;
    seed.run(images[choice], test);
    for(float i:test){
        std::cout<<i<<"\n";
    }
    print_convolution(seed,images[choice]);
    //print_pool(seed,images[0]);
    evolve(seed,1000,80,best);
    float res;
    evaluate(seed,labels,images,10000,0,&res,false);
    std::cout<<"Original Network Accuracy: "<<res<<"%\n";
    evaluate(*best,labels,images,10000,0,&res,false);
    std::cout<<"Final Network Accuracy: "<<res<<"%";

   	return 0;	
}
