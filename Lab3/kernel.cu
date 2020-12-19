
#include "cuda_runtime.h"
#include <cuda.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "nlohmann/json.hpp"
#include "device_launch_parameters.h"

using namespace std;
using json = nlohmann::json;

const int MAX_STRING_LENGTH = 256;
const int THREADS = 3;
#define DATA_FILE "IFF-7_2_MilisiunasJ_L1_dat_1.json"
#define RESULTS_FILE "a.json"

struct Car {
    char brand[MAX_STRING_LENGTH];
    int make_year;
    double mileage;

    void from_json(json data) {
        string brand_temp = data["brand"];
        memset(brand, 0, sizeof(brand));
        strcpy(brand, brand_temp.c_str());

        make_year = data["makeYear"];
        mileage = data["mileage"];
    }

    void print() {
        printf("Brand: %s Make Year: %d Mileage: %f\n", brand, make_year, mileage);
    }
};

void readCarsFile(vector<Car>* cars);
void write_results_to_file(Car* cars, int n, const string file_path, const string title);

__global__ void sum_on_gpu(Car* cars, int* n, int* chunk_size, Car* results);
__device__ void gpu_memset(char* dest);
__device__ void gpu_strcat(char* dest, char* src, int filter);
__device__ char gpu_filter(int src);
char* gpu_convertIntegerToChar(int N);
int main() {
    vector<Car> all_cars;
    readCarsFile(&all_cars);


    printf("%x", gpu_convertIntegerToChar(2015));

    // Host
    Car* cars = &all_cars[0];
    Car results[THREADS];
    int n = all_cars.size();
    int chunk_size = n / THREADS;

    // GPU
    Car* d_all_cars;
    int* d_n;
    int* d_chunk_size;
    Car* d_results;

    // Memory allocation for GPU
    cudaMalloc((void**)&d_all_cars, n * sizeof(Car));
    cudaMalloc((void**)&d_results, THREADS * sizeof(Car));
    cudaMalloc((void**)&d_n, sizeof(int));
    cudaMalloc((void**)&d_chunk_size, sizeof(int));

    // Copies memory from CPU to GPU
    cudaMemcpy(d_all_cars, cars, n * sizeof(Car), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_chunk_size, &chunk_size, sizeof(int), cudaMemcpyHostToDevice);

    sum_on_gpu <<< 1, THREADS >>> (d_all_cars, d_n, d_chunk_size, d_results);
    printf("\nKiekvienos gijos susumuoti duomenys:\n");
    cudaDeviceSynchronize();

    cudaMemcpy(&results, d_results, THREADS * sizeof(Car), cudaMemcpyDeviceToHost);
    cudaFree(d_all_cars);
    cudaFree(d_n);
    cudaFree(d_chunk_size);
    cudaFree(d_results);

    cout << "Finished" << endl;
    write_results_to_file(results, THREADS, RESULTS_FILE, "A dalies rezultatai");
    return 0;
}

/**
 * GPU
 * Sums cars list chunk data properties
 * @param cars Cars list
 * @param n Cars list size
 * @param chunk_size Summed items per thread
 * @param results Summed chunk results
 */
__global__ void sum_on_gpu(Car* cars, int* n, int* chunk_size, Car* results) {
    int start_index = threadIdx.x * *chunk_size;
    int end_index = min(start_index + *chunk_size, *n);

    if (end_index + *chunk_size >= *n || threadIdx.x + 1 == THREADS) {
        end_index = *n;
    }

    Car sum;
    gpu_memset(sum.brand);
    sum.make_year = 0;
    sum.mileage = 0.0;

    for (int i = start_index; i < end_index; ++i) {
        if (i < *n) {
            gpu_strcat(sum.brand, cars[i].brand, cars[i].make_year);
            sum.make_year += cars[i].make_year;
            sum.mileage += cars[i].mileage;
        }
    }

    //    printf("Thread: %d Start Index: %d End Index: %d\n", threadIdx.x, start_index, end_index);
    results[threadIdx.x] = sum;
    printf("Thread: %d Brand: %s Make Year: %d Mileage: %f\n", threadIdx.x, results[threadIdx.x].brand, results[threadIdx.x].make_year, results[threadIdx.x].mileage);
}

/**
 * Appends char array to other char array
 * @param dest Destination array
 * @param src Source array
 */
__device__ void gpu_strcat(char* dest, char* src, int filter) {
    for (int i = 0; i < MAX_STRING_LENGTH; ++i) {
        if (dest[i] == 0) {
            for (int j = 0; j < MAX_STRING_LENGTH; ++j) {
                if (src[j] != 0) {
                    dest[i + j] = src[j];
                }
                else {
                    dest[i + j] = '-';
                    dest[i + j + 1] = gpu_filter(filter);
                    dest[i + j + 2] = ' ';
                    dest[i + j + 3] = ' ';
                    dest[i + j + 4] = ' ';
                    break;
                }
            }
            break;
        }
    }
}


__device__ char gpu_filter(int src) {

    char filter;
    if (src > 2015) {
        filter = 'A';
    }
    else {
        filter = 'B';
    }
    return filter;
}




/**
 * Zeroes all char memory
 * @param dest Char array
 */
__device__ void gpu_memset(char* dest) {
    for (int i = 0; i < MAX_STRING_LENGTH; ++i) {
        dest[i] = 0;
    }
}




// Function to convert integer to
// character array
char* gpu_convertIntegerToChar(int N)
{

    char result[sizeof(N)];


    int tmp;
    int p = 0;
    tmp = N;
        while (tmp != 0)
        {
            result[p] = (tmp % 10);
            tmp = (tmp - result[p]) / 10;
            p++;
        }
        return result;

}






/**
 * Reads cards data in isjon format from the file,
 * parses to Car structs
 * @param cars Destination list
 */
void readCarsFile(vector<Car>* cars) {
    ifstream stream(DATA_FILE);
    json allCarsJson = json::parse(stream);
    auto allCars = allCarsJson["cars"];
    printf("Pradiniai duomenys:\n");
    for (const json& new_car : allCars) {
        Car tempCar;
        tempCar.from_json(new_car);
        tempCar.print();
        cars->push_back(tempCar);
    }
}

/**
 * Writes given monitor cars formatted in table to file
 * @param cars Cars list
 * @param file_path Result file path
 * @param title Results table title
 */
void write_results_to_file(Car* cars, int n, const string file_path, const string title) {
    ofstream file;
    file.open(file_path);
    file << setw(70) << title << endl
        << "------------------------------------------------------------------------------------------------------------------------"
        << endl
        << setw(60) << "Gamintojas |" << setw(30) << "Pagaminimo metai |" << setw(30) << "Rida" << endl
        << "------------------------------------------------------------------------------------------------------------------------"
        << endl;
    for (int i = 0; i < n; ++i) {
        string str(cars[i].brand);
        file << setw(60) << str + " |" << setw(30) << to_string(cars[i].make_year) + " |"
            << setw(30)
            << to_string(cars[i].mileage) << endl;
    }

    file << endl << endl << endl;
}