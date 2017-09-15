#ifndef UTILS_H
#define UTILS_H

#include <cublas_v2.h>
#include <host_defines.h>
#include <thrust/random.h>

#define gpuErrorCheckCuda(ans) { Utility::gpuAssert((ans), __FILE__, __LINE__); }
#define gpuErrorCheckCublas(ans) { Utility::gpuAssertCublas((ans), __FILE__, __LINE__); }


#define MEX
#ifdef MEX
	#include "mex.h"
	#define printf mexPrintf
#endif

namespace Utility {
	const char* cublasGetErrorString(cublasStatus_t status);

	template<typename T>
	inline void printArrayCPU(const char* msg, const T* const memory, const int length) {
		printf("%s:", msg);
		for(int i = 0; i < length; i++)
			printf("%.10f ", memory[i]);
		printf("\n");
	}

	template<typename T>
	inline void printArrayGPU(const char* msg, const T* const deviceMemory, const int length) {
		T* value = new T[length];	
		cudaMemcpy(value, deviceMemory, sizeof(T)*length, cudaMemcpyDeviceToHost);
		printf("%s:", msg);
		for(int i = 0; i < length; i++)
			printf("%.10f ", value[i]);
		printf("\n");
		delete[] value;
	}

	inline void gpuAssert(cudaError_t code, const char *file, int line) {
	  	if (code != cudaSuccess)
			printf("GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
	}

	inline void gpuAssertCublas(cublasStatus_t code, const char *file, int line) {
	  	if (code != CUBLAS_STATUS_SUCCESS) 
			printf("GPU (CUBLAS) assert: %s %s %d\n", cublasGetErrorString(code), file, line);
	}
	
	inline __device__  double sigmoidFunction(double value) { return 1 / (1 + exp(-value)); }
	inline __device__ double reluFunction(double value) { return value > 0 ? value : 0; }
	
	struct GeneratorUniform {
		float a, b;
		inline __host__ __device__ GeneratorUniform(float a_, float b_) : a(a_), b(b_) {}
	    	inline __host__ __device__ float operator () (const unsigned int n) const {
			thrust::default_random_engine rng;
			thrust::uniform_real_distribution<float> dist(a, b);
			rng.discard(n);
			return dist(rng);
		}

	};

	struct GeneratorNormal {
		float mean, std;
		inline __host__ __device__ GeneratorNormal(float mean_, float std_) : mean(mean_), std(std_) {}
	    	inline __host__ __device__ float operator () (const unsigned int n) const {
			thrust::default_random_engine rng;
			thrust::random::normal_distribution<float> dist(mean, std);
			rng.discard(n);
			return dist(rng);
		}

	};

	struct Gradients {
		double* deviceVisiblePositiveGradients;
		double* deviceHiddenPositiveGradients;
		double* deviceVisibleNegativeGradients;
		double* deviceHiddenNegativeGradients;
		Gradients(double* deviceVisiblePositiveGradients_, double* deviceHiddenPositiveGradients_, double* deviceVisibleNegativeGradients_, double* deviceHiddenNegativeGradients_);
		~Gradients();
	};

	struct Deltas {
		double* deviceDeltaWeights;
		double* deviceDeltaBiasesVisible;
		double* deviceDeltaBiasesHidden;
		Deltas(double* deviceDeltaWeightsWeights_, double* deviceDeltaBiasesVisible_, double* deviceDeltaBiasesHidden_);
		~Deltas();
	};
}

#endif

