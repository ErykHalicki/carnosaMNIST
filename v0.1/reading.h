#include <string>
unsigned char** read_mnist_images(std::string full_path, int number_of_images, int image_size);
unsigned char* read_mnist_labels(std::string full_path, int number_of_labels);
