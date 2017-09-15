#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "Utility.cuh"


namespace Utility {
	const char* cublasGetErrorString(cublasStatus_t status)
	{
	    switch(status)
	    {
		case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
		case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
		case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
		case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
		case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
		case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
		case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
		case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
	    }
	    return "Unknown cuBLAS error";
	}

	Gradients::Gradients(double* deviceVisiblePositiveGradients_, double* deviceHiddenPositiveGradients_, double* deviceVisibleNegativeGradients_, double* deviceHiddenNegativeGradients_) :
		deviceVisiblePositiveGradients(deviceVisiblePositiveGradients_), 
		deviceHiddenPositiveGradients(deviceHiddenPositiveGradients_), 
		deviceVisibleNegativeGradients(deviceVisibleNegativeGradients_), 
		deviceHiddenNegativeGradients(deviceHiddenNegativeGradients_) {}

	Gradients::~Gradients() {
		cudaFree(deviceHiddenPositiveGradients);
		cudaFree(deviceVisibleNegativeGradients);
		cudaFree(deviceHiddenNegativeGradients);
	}

	Deltas::Deltas(double* deviceDeltaWeights_, double* deviceDeltaBiasesVisible_, double* deviceDeltaBiasesHidden_) :
		deviceDeltaWeights(deviceDeltaWeights_), 
		deviceDeltaBiasesVisible(deviceDeltaBiasesVisible_), 
		deviceDeltaBiasesHidden(deviceDeltaBiasesHidden_) {}

	Deltas::~Deltas() {
		cudaFree(deviceDeltaWeights);
		cudaFree(deviceDeltaBiasesVisible);
		cudaFree(deviceDeltaBiasesHidden);
	}
}
