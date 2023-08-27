#ifndef WRITING_H
#include <string>
#include <iostream>
#include <fstream>
static std::ofstream out;
void write_data(std::string data);
void add_data(std::string* data, int num);
void add_data(std::string* data, float num);
void start_data(std::string name);
void end_data();
#endif
