#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

using json = nlohmann::json;
using namespace std;

// myFunction<<<block_count, thread_count>>>();

// __global__ - functions that can be called from CPU (host) and be executed on GPU (device)
// __device__ - functions that can be called and executed from GPU (Device)

// Constants FOR CPU
const int INPUT_FILE_DATA_COUNT = 300;
const int WORD_LENGTH = 25;

// Constats FOR GPU
__device__ const int GPU_WORD_LENGTH = 25;

class Anime{
public:
    char Name[WORD_LENGTH];
    int ReleaseYear;
    double Score;
};

__device__ int HashFunction(int length) {
    int hash = length; // Start with the input value
    hash = (hash * 31) ^ (hash >> 3); // Multiply and XOR with a shifted value
    hash = (hash * 17) + 12345;       // Mix it further with another constant
    hash = hash % 5000;         // Ensure it fits into [0, 4999]
    printf("%d\n", hash);
    return hash;
}

__global__ void CudaCalculations(const Anime* anime, char* resultCharArray, const int* inputDataLength, int* resultIndexCount) {
    int arrayIndex = blockIdx.x * blockDim.x + threadIdx.x;

    //Check if operation is outside of array
    if (arrayIndex >= *inputDataLength)
        return;

    Anime element = anime[arrayIndex];
    int hash = HashFunction(element.Score);
    

    //Filter check, if fail = not added
    if (hash >= 2500)
        return;

    int index = atomicAdd(resultIndexCount, 1); //Adds a word
    index = index * GPU_WORD_LENGTH; //Recalculate the next word start

    for (int charIndex = 0; element.Name[charIndex] != 0; charIndex++, index++) {
        resultCharArray[index] = element.Name[charIndex];
    }

    //Add a delimiter
    resultCharArray[index] = '-';
    index++;

    //Add last range
    if (hash >= 1250) {
        resultCharArray[index++] = '1';
        resultCharArray[index++] = '2';
        resultCharArray[index++] = '5';
        resultCharArray[index++] = '0';
        resultCharArray[index++] = '>';
        resultCharArray[index++] = '=';
    }
    else {
        resultCharArray[index++] = '1';
        resultCharArray[index++] = '2';
        resultCharArray[index++] = '5';
        resultCharArray[index++] = '0';
        resultCharArray[index++] = '<';
    }
}

int main() {
    printf("Program started\n");

    string inputFileName = "Algimantas_Vezevicius_data_3.json";
    string outputFileName = "Algimantas_Vezevicius_rez_3.txt";

    //reading
    ifstream inputFile(inputFileName);
    json jsonData;
    inputFile >> jsonData;

    const int inputDataCount = jsonData["AnimeData"].size();
    cout << "Read Data Size: " << inputDataCount << endl;

    Anime anime[INPUT_FILE_DATA_COUNT];
    for (int i = 0; i < jsonData["AnimeData"].size(); i++) {
        json currentJson = jsonData["AnimeData"][i];

        string name = currentJson["Name"];
        int year = currentJson["ReleaseYear"];
        double rating = currentJson["Rating"];

        anime[i].ReleaseYear = year;
        anime[i].Score = rating;
        strcpy(anime[i].Name, name.c_str());
    }
    printf("Data reading finished\n");

    //ponters for memory transfer for CUDA to VRAM - CUDA
    int resultCounter = 0;
    char* gpu_result;
    int* gpu_resultCounter;
    int* gpu_inputSizeCount;
    Anime* gpu_inputData;

    //Memory allocation to GPU VRAM
    cudaMalloc(&gpu_inputData, sizeof(anime));
    cudaMalloc(&gpu_result, sizeof(char) * WORD_LENGTH * INPUT_FILE_DATA_COUNT);
    cudaMalloc(&gpu_resultCounter, sizeof(int));
    cudaMalloc(&gpu_inputSizeCount, sizeof(int));

    //CPU RAM to GPU VRAM copy allocation
    cudaMemcpy(gpu_inputData, anime, sizeof(anime), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_resultCounter, &resultCounter, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_inputSizeCount, &INPUT_FILE_DATA_COUNT, sizeof(int), cudaMemcpyHostToDevice);

    //Optimal block and thread size calculation --- CUDA
    int thread_count = 32 * 2;
    int block_count = (inputDataCount / thread_count) + 1;
    int outputDataCount = thread_count * block_count;
    cout << "Thread Count: " << thread_count << endl;
    cout << "Block Count: " << block_count << endl;
    cout << "End Array Size: " << outputDataCount << endl;

    //GPU work --- CUDA
    CudaCalculations <<<block_count, thread_count>>> (gpu_inputData, gpu_result, gpu_inputSizeCount, gpu_resultCounter);
    cudaDeviceSynchronize();

    //Copying GPU VRAM to CPU RAM --- CUDA
    char results[WORD_LENGTH * INPUT_FILE_DATA_COUNT];
    cudaMemcpy(results, gpu_result, sizeof(results), cudaMemcpyDeviceToHost);
    cudaMemcpy(&resultCounter, gpu_resultCounter, sizeof(int), cudaMemcpyDeviceToHost);

    //Memory release so to not cause a leak --- CUDA
    cudaFree(gpu_result);
    cudaFree(gpu_resultCounter);
    cudaFree(gpu_inputData);

    //Results
    cout << "Result count: " << resultCounter << endl;
    ofstream outputFile(outputFileName);
    for (int i = 0; i < WORD_LENGTH * resultCounter; i++) {
        if ((i + 1) % WORD_LENGTH == 0) //end of word
            outputFile << endl;
        else
            outputFile << results[i];
    }

    outputFile.close();

    printf("Program finished\n");
    return 0;
}
