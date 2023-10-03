#include <string>
#include "writing.h"
#include <fstream>
#include <iostream>

void add_data(std::string* data, float num){
    *data +=std::to_string(num);
    *data +=",";
}
void add_data(std::string* data, int num){
    *data+=std::to_string(num);
    *data+=",";
}

void write_data(std::string data){
    data+="\n";
    out << data;
}
void start_data(std::string name){
    out=std::ofstream(name);
}
void end_data(){
    out.close();
}
