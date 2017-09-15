#include <cmath>
#include <string.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include "Utility.cuh"
#include "RestrictedBoltzmannMachine.cuh"


//#define DEBUG
#define MEX

#ifdef MEX
	#include "mex.h"
	#define printf mexPrintf
#endif


//-----BinaryBinaryRestrictedBoltzmannMachine-----

BinaryBinaryRestrictedBoltzmannMachine::BinaryBinaryRestrictedBoltzmannMachine(const double* const weights_, const double* const biasesVisible_, const double* const biasesHidden_, const int numHidden_, const int numVisible_): RestrictedBoltzmannMachine(weights_, biasesVisible_, biasesHidden_, numHidden_, numVisible_) { }

BinaryBinaryRestrictedBoltzmannMachine::~BinaryBinaryRestrictedBoltzmannMachine() { }


void BinaryBinaryRestrictedBoltzmannMachine::HiddenTransformation(double* const deviceHiddenMatrix, const int samples) const {
	// Sigmoid
	const auto deviceIt = thrust::device_pointer_cast(deviceHiddenMatrix);
	thrust::transform(deviceIt, deviceIt + samples*numHidden + 1, deviceIt, [] __device__ (double x) { return Utility::sigmoidFunction(x); });

	// Sampling	
	int seed = rand();
	thrust::device_vector<float> deviceRandomVector(samples * numHidden);
	thrust::counting_iterator<unsigned int> indexSequenceBegin(seed);
	thrust::transform(thrust::device, indexSequenceBegin, indexSequenceBegin + samples*numHidden + 1, deviceRandomVector.begin(), Utility::GeneratorUniform(0.0f, 1.0f));
	#ifdef DEBUG
		float* deviceRandomVectorPtr = thrust::raw_pointer_cast(deviceRandomVector.data());
		Utility::printArrayGPU("Hidden uniform random", deviceRandomVectorPtr, samples*numHidden);
	#endif
	thrust::transform(deviceIt, deviceIt + samples*numHidden + 1, deviceRandomVector.begin(), deviceIt, [] __device__ (double x, double y) { return x > y ? 1.0f : 0.0f; });
}

void BinaryBinaryRestrictedBoltzmannMachine::VisibleTransformation(double* const deviceVisibleMatrix, const int samples) const {
	// Sigmoid	
	const auto deviceIt = thrust::device_pointer_cast(deviceVisibleMatrix);	
	thrust::transform(deviceIt, deviceIt + samples*numVisible + 1, deviceIt, [] __device__ (double x) { return Utility::sigmoidFunction(x); });

	// Sampling	
	int seed = rand();
	thrust::device_vector<float> deviceRandomVector(samples * numVisible);
	thrust::counting_iterator<unsigned int> indexSequenceBegin(seed);
	thrust::transform(thrust::device, indexSequenceBegin, indexSequenceBegin + samples*numVisible + 1, deviceRandomVector.begin(), Utility::GeneratorUniform(0.0f, 1.0f));
	#ifdef DEBUG
		float* deviceRandomVectorPtr = thrust::raw_pointer_cast(deviceRandomVector.data());
		Utility::printArrayGPU("Visible uniform random", deviceRandomVectorPtr, samples*numVisible);
	#endif
	thrust::transform(deviceIt, deviceIt + samples*numVisible + 1, deviceRandomVector.begin(), deviceIt, [] __device__ (double x, double y) { return x > y ? 1.0f : 0.0f; });
}

//-----BinaryBinaryRestrictedBoltzmannMachine-----


//-----LinearNReluRestrictedBoltzmannMachine-----

LinearNReluRestrictedBoltzmannMachine::LinearNReluRestrictedBoltzmannMachine(const double* const weights_, const double* const biasesVisible_, const double* const biasesHidden_, const int numHidden_, const int numVisible_): RestrictedBoltzmannMachine(weights_, biasesVisible_, biasesHidden_, numHidden_, numVisible_) { }

LinearNReluRestrictedBoltzmannMachine::~LinearNReluRestrictedBoltzmannMachine() { }


void LinearNReluRestrictedBoltzmannMachine::HiddenTransformation(double* const deviceHiddenMatrix, const int samples) const {
	#ifdef DEBUG
		Utility::printArrayGPU("Hidden inputs states:", deviceHiddenMatrix, samples*numHidden);
	#endif	

	// Generate normal noise	
	int seed = rand();
	thrust::device_vector<float> deviceRandomVector(samples*numHidden);
	thrust::counting_iterator<unsigned int> indexSequenceBegin(seed);
	thrust::transform(thrust::device, indexSequenceBegin, indexSequenceBegin + samples*numHidden + 1, deviceRandomVector.begin(), Utility::GeneratorNormal(0.0f, 1.0f));
	#ifdef DEBUG
		float* deviceRandomVectorPtr = thrust::raw_pointer_cast(deviceRandomVector.data());
		Utility::printArrayGPU("Hidden normal random", deviceRandomVectorPtr, samples*numHidden);
	#endif
	
	// Add normal noise
	const auto deviceIt = thrust::device_pointer_cast(deviceHiddenMatrix);
	thrust::transform(deviceIt, deviceIt + samples*numHidden + 1, deviceRandomVector.begin(), deviceIt, thrust::plus<double>());
	#ifdef DEBUG
		Utility::printArrayGPU("Hidden states with noise", deviceHiddenMatrix, samples*numHidden);
	#endif
	
	// Max pooling
	thrust::transform(deviceIt, deviceIt + samples*numHidden + 1, deviceIt, [] __device__ (double x) { return x > 0 ? x : 0; });
	#ifdef DEBUG
		Utility::printArrayGPU("Hidden states with ReLU", deviceHiddenMatrix, samples*numHidden);
	#endif
}

void LinearNReluRestrictedBoltzmannMachine::VisibleTransformation(double* const deviceVisibleMatrix, const int samples) const {
	#ifdef DEBUG
		Utility::printArrayGPU("Visible inputs states:", deviceVisibleMatrix, samples*numVisible);
	#endif
}



//-----LinearNReluRestrictedBoltzmannMachine-----


//-----LinearBinaryRestrictedBoltzmannMachine-----

LinearBinaryRestrictedBoltzmannMachine::LinearBinaryRestrictedBoltzmannMachine(const double* const weights_, const double* const biasesVisible_, const double* const biasesHidden_, const int numHidden_, const int numVisible_): RestrictedBoltzmannMachine(weights_, biasesVisible_, biasesHidden_, numHidden_, numVisible_) { }

LinearBinaryRestrictedBoltzmannMachine::~LinearBinaryRestrictedBoltzmannMachine() { }


void LinearBinaryRestrictedBoltzmannMachine::HiddenTransformation(double* const deviceHiddenMatrix, const int samples) const {
	// Sigmoid
	const auto deviceIt = thrust::device_pointer_cast(deviceHiddenMatrix);
	thrust::transform(deviceIt, deviceIt + samples*numHidden + 1, deviceIt, [] __device__ (double x) { return Utility::sigmoidFunction(x); });
	// Sampling	
	int seed = rand();
	thrust::device_vector<float> deviceRandomVector(samples*numHidden);
	thrust::counting_iterator<unsigned int> indexSequenceBegin(seed);
	thrust::transform(thrust::device, indexSequenceBegin, indexSequenceBegin + samples*numHidden + 1, deviceRandomVector.begin(), Utility::GeneratorUniform(0.0f, 1.0f));
	#ifdef DEBUG
		float* deviceRandomVectorPtr = thrust::raw_pointer_cast(deviceRandomVector.data());
		Utility::printArrayGPU("Hidden uniform random", deviceRandomVectorPtr, samples*numHidden);
	#endif
	thrust::transform(deviceIt, deviceIt + samples*numHidden + 1, deviceRandomVector.begin(), deviceIt, [] __device__ (double x, double y) { return x > y ? 1 : 0; });
}

void LinearBinaryRestrictedBoltzmannMachine::VisibleTransformation(double* const deviceVisibleMatrix, const int samples) const { }

//-----LinearBinaryRestrictedBoltzmannMachine-----


RestrictedBoltzmannMachine::RestrictedBoltzmannMachine(const double* const weights_, const double* const biasesVisible_, const double* const biasesHidden_, const int numHidden_, const int numVisible_) {
	srand (10);	
	numVisible = numVisible_;
	numHidden = numHidden_;

	// Allocate memory
	weights = new double[numHidden*numVisible]();
	biasesVisible = new double[numVisible]();
	biasesHidden = new double[numHidden]();
	gpuErrorCheckCuda(cudaMalloc(&deviceWeights, sizeof(double)*numVisible*numHidden));
	gpuErrorCheckCuda(cudaMalloc(&deviceBiasesHidden, sizeof(double)*numHidden));
	gpuErrorCheckCuda(cudaMalloc(&deviceBiasesVisible, sizeof(double)*numVisible));

	// Weights initialization
	for (int i = 0; i < numHidden*numVisible; i++) {
		weights[i] = weights_[i];
	        if (i < numHidden)	
            		biasesHidden[i] = biasesHidden_[i];
	        if (i < numVisible)
        		biasesVisible[i] = biasesVisible_[i];
   	}

		
	// Copy data to a device
	gpuErrorCheckCuda(cudaMemcpy(deviceWeights, weights, sizeof(double)*numVisible*numHidden, cudaMemcpyHostToDevice));
	gpuErrorCheckCuda(cudaMemcpy(deviceBiasesVisible, biasesVisible, sizeof(double)*numVisible, cudaMemcpyHostToDevice));
	gpuErrorCheckCuda(cudaMemcpy(deviceBiasesHidden, biasesHidden, sizeof(double)*numHidden, cudaMemcpyHostToDevice));

	// Handle creation
	gpuErrorCheckCublas(cublasCreate(&defaultHandle));
}

RestrictedBoltzmannMachine::~RestrictedBoltzmannMachine() {
	delete[] weights;
	delete[] biasesVisible;
	delete[] biasesHidden;
	gpuErrorCheckCuda(cudaFree(deviceWeights));
	gpuErrorCheckCuda(cudaFree(deviceBiasesVisible));
	gpuErrorCheckCuda(cudaFree(deviceBiasesHidden));
	gpuErrorCheckCublas(cublasDestroy(defaultHandle));
}

void RestrictedBoltzmannMachine::GetHiddenGivenVisible(double* const deviceHiddenMatrix, const double* const deviceVisibleMatrix, const double* const deviceResizedBiasesHidden, const int samples) const {
	const double alpha = 1.0;
	const double beta = 0.0;
	gpuErrorCheckCublas(cublasDgemm(defaultHandle, CUBLAS_OP_N, CUBLAS_OP_T, samples, numHidden, numVisible, &alpha, deviceVisibleMatrix, samples, deviceWeights, numHidden, &beta, deviceHiddenMatrix, samples));
	gpuErrorCheckCublas(cublasDaxpy(defaultHandle, samples*numHidden, &alpha, deviceResizedBiasesHidden, 1, deviceHiddenMatrix, 1));
	HiddenTransformation(deviceHiddenMatrix, samples);
}

void RestrictedBoltzmannMachine::GetVisibleGivenHidden(double* const deviceVisibleMatrix, const double* const deviceHiddenMatrix, const double* const deviceResizedBiasesVisible, const int samples) const {
	const double alpha = 1.0;
	const double beta = 0.0;
	gpuErrorCheckCublas(cublasDgemm(defaultHandle, CUBLAS_OP_N, CUBLAS_OP_N, samples, numVisible, numHidden, &alpha, deviceHiddenMatrix, samples, deviceWeights, numHidden, &beta, deviceVisibleMatrix, samples));
	gpuErrorCheckCublas(cublasDaxpy(defaultHandle, samples*numVisible, &alpha, deviceResizedBiasesVisible, 1, deviceVisibleMatrix, 1));
	VisibleTransformation(deviceVisibleMatrix, samples);
}

double* RestrictedBoltzmannMachine::GibbsChainIteration(const double* const data, const int samples, int steps, const char* mode) const {
	if (!strcmp(mode, "Reconstruction"))
		steps = 1;
		
	// Device memory allocation and copying
	double* deviceData; gpuErrorCheckCuda(cudaMalloc(&deviceData, sizeof(double)*numVisible*samples));
	gpuErrorCheckCuda(cudaMemcpy(deviceData, data, sizeof(double)*numVisible*samples, cudaMemcpyHostToDevice));
	
	// Gibbs chain
	Utility::Gradients g = ContrastiveDivergence(deviceData, samples, steps);

	// Copy results
	double* result;	
	if (!strcmp(mode, "Sample")) {
		result = new double[numVisible*samples];
		gpuErrorCheckCuda(cudaMemcpy(result, g.deviceVisibleNegativeGradients, sizeof(double)*numVisible*samples, cudaMemcpyDeviceToHost));
	} else if (!strcmp(mode, "Reconstruction")) {
		result = new double[numHidden*samples];
		gpuErrorCheckCuda(cudaMemcpy(result, g.deviceHiddenPositiveGradients, sizeof(double)*numHidden*samples, cudaMemcpyDeviceToHost));
	} else printf("You should pass a value from {Sample, Reconstruction} to result parameter in GibbsChainIteration.");

	// Free memory
	gpuErrorCheckCuda(cudaFree(deviceData));
	return result;
}

Utility::Gradients RestrictedBoltzmannMachine::ContrastiveDivergence(double* const deviceData, const int samples, const int steps) const {
	// Memory allocation
	double* deviceResizedBiasesHidden; gpuErrorCheckCuda(cudaMalloc(&deviceResizedBiasesHidden, sizeof(double)*samples*numHidden));
	double* deviceResizedBiasesVisible; gpuErrorCheckCuda(cudaMalloc(&deviceResizedBiasesVisible, sizeof(double)*samples*numVisible));
	double* devicePositiveVisible; gpuErrorCheckCuda(cudaMalloc(&devicePositiveVisible, sizeof(double)*samples*numVisible));
	double* devicePositiveHidden; gpuErrorCheckCuda(cudaMalloc(&devicePositiveHidden, sizeof(double)*samples*numHidden));
	double* deviceNegativeVisible; gpuErrorCheckCuda(cudaMalloc(&deviceNegativeVisible, sizeof(double)*samples*numVisible));
	double* deviceNegativeHidden; gpuErrorCheckCuda(cudaMalloc(&deviceNegativeHidden, sizeof(double)*samples*numHidden));
	
	
	// Data referencing
	devicePositiveVisible = deviceData;

	// Data resizing for biases
	for (int i = 0; i < numHidden; i++) 
		gpuErrorCheckCublas(cublasDcopy(defaultHandle, samples, deviceBiasesHidden + i, 0, deviceResizedBiasesHidden + i*samples, 1));
	
	for (int i = 0; i < numVisible; i++) 
		gpuErrorCheckCublas(cublasDcopy(defaultHandle, samples, deviceBiasesVisible + i, 0, deviceResizedBiasesVisible + i*samples, 1));
	
	// Gibbs chain
	GetHiddenGivenVisible(devicePositiveHidden, devicePositiveVisible, deviceResizedBiasesHidden, samples);
	for (int i = 0; i < steps; i++) {
		GetVisibleGivenHidden(deviceNegativeVisible, i == 0 ? devicePositiveHidden : deviceNegativeHidden, deviceResizedBiasesVisible, samples);
		GetHiddenGivenVisible(deviceNegativeHidden, deviceNegativeVisible, deviceResizedBiasesHidden, samples);
		#ifdef DEBUG	
			Utility::printArrayGPU("positiveVisible", devicePositiveVisible, samples*numVisible);
			Utility::printArrayGPU("positiveHidden", devicePositiveHidden, samples*numHidden);
			Utility::printArrayGPU("negativeVisible", deviceNegativeVisible, samples*numVisible);
			Utility::printArrayGPU("negativeHidden", deviceNegativeHidden, samples*numHidden);
		#endif
	}

	// Memory deallocation
	gpuErrorCheckCuda(cudaFree(deviceResizedBiasesHidden));
	gpuErrorCheckCuda(cudaFree(deviceResizedBiasesVisible));


	// Results
	return Utility::Gradients(devicePositiveVisible, devicePositiveHidden, deviceNegativeVisible, deviceNegativeHidden);
}




Utility::Deltas RestrictedBoltzmannMachine::GetDeltaWeights(Utility::Gradients& gradients, const int samples, const double learningRate) const {
	// Memory allocation
	double* deviceDeltaweights1; gpuErrorCheckCuda(cudaMalloc(&deviceDeltaweights1, sizeof(double)*numHidden*numVisible));
	double* deviceDeltaweights2; gpuErrorCheckCuda(cudaMalloc(&deviceDeltaweights2, sizeof(double)*numHidden*numVisible));
	double* deviceDeltaBiasesVisible; gpuErrorCheckCuda(cudaMalloc(&deviceDeltaBiasesVisible, sizeof(double)*numVisible));
	double* deviceDeltaBiasesHidden; gpuErrorCheckCuda(cudaMalloc(&deviceDeltaBiasesHidden, sizeof(double)*numHidden));
	thrust::device_ptr<double> deviceIt;
	deviceIt = thrust::device_pointer_cast(deviceDeltaweights1); thrust::fill_n(deviceIt, numHidden*numVisible + 1, 0.0);
	deviceIt = thrust::device_pointer_cast(deviceDeltaweights2); thrust::fill_n(deviceIt, numHidden*numVisible + 1, 0.0);
	deviceIt = thrust::device_pointer_cast(deviceDeltaBiasesVisible); thrust::fill_n(deviceIt, numVisible + 1, 0.0);
	deviceIt = thrust::device_pointer_cast(deviceDeltaBiasesHidden); thrust::fill_n(deviceIt, numHidden + 1, 0.0);

	// Weights changing
	const double alpha = learningRate / samples;
	for (int i = 0; i < samples; i++) {
		double* v = gradients.deviceVisiblePositiveGradients + i;
		double* h = gradients.deviceHiddenPositiveGradients + i;
		double* vprim = gradients.deviceVisibleNegativeGradients + i;
		double* hprim = gradients.deviceHiddenNegativeGradients + i;
		
		gpuErrorCheckCublas(cublasDger(defaultHandle, numHidden, numVisible, &alpha, h, samples, v, samples, deviceDeltaweights1, numHidden));
		gpuErrorCheckCublas(cublasDger(defaultHandle, numHidden, numVisible, &alpha, hprim, samples, vprim, samples, deviceDeltaweights2, numHidden));
	}
	#ifdef DEBUG	
		Utility::printArrayGPU("deviceDeltaweights1", deviceDeltaweights1, numHidden*numVisible);
		Utility::printArrayGPU("deviceDeltaweights2", deviceDeltaweights2, numHidden*numVisible);
	#endif
	
	// Substract weights
	const double minusOne = -1.0;
	gpuErrorCheckCublas(cublasDaxpy(defaultHandle, numHidden*numVisible, &minusOne, deviceDeltaweights2, 1, deviceDeltaweights1, 1));
	gpuErrorCheckCuda(cudaFree(deviceDeltaweights2));

	// Additional memory allocation for biases
	double* deviceOnes; gpuErrorCheckCuda(cudaMalloc(&deviceOnes, sizeof(double)*samples));
	deviceIt = thrust::device_pointer_cast(deviceOnes); thrust::fill_n(deviceIt, samples + 1, 1.0);

	const double beta = 0.0;
	// Visible biases
	gpuErrorCheckCublas(cublasDaxpy(defaultHandle, samples*numVisible, &minusOne, gradients.deviceVisibleNegativeGradients, 1, gradients.deviceVisiblePositiveGradients, 1));
	gpuErrorCheckCublas(cublasDgemv(defaultHandle, CUBLAS_OP_T, samples, numVisible, &alpha, gradients.deviceVisiblePositiveGradients, samples, deviceOnes, 1, &beta, deviceDeltaBiasesVisible, 1));
	
	// Hidden biases
	gpuErrorCheckCublas(cublasDaxpy(defaultHandle, samples*numHidden, &minusOne, gradients.deviceHiddenNegativeGradients, 1, gradients.deviceHiddenPositiveGradients, 1));
	gpuErrorCheckCublas(cublasDgemv(defaultHandle, CUBLAS_OP_T, samples, numHidden, &alpha, gradients.deviceHiddenPositiveGradients, samples, deviceOnes, 1, &beta, deviceDeltaBiasesHidden, 1));

	#ifdef DEBUG	
		Utility::printArrayGPU("deviceDeltaBiasesVisible", deviceDeltaBiasesVisible, numVisible);
		Utility::printArrayGPU("deviceDeltaBiasesVisible", deviceDeltaBiasesHidden, numHidden);
	#endif

	// Freeing additional memory
	gpuErrorCheckCuda(cudaFree(deviceOnes));

	// Results
	return Utility::Deltas(deviceDeltaweights1, deviceDeltaBiasesVisible, deviceDeltaBiasesHidden);
}

Utility::Deltas RestrictedBoltzmannMachine::TrainingIteration(double* const deviceData, const int samples, const double learningRate, const int steps) const {
	Utility::Gradients g = ContrastiveDivergence(deviceData, samples, steps);
	return GetDeltaWeights(g, samples, learningRate);
}

void RestrictedBoltzmannMachine::TrainNetwork(const double* const data, const int samples, int miniBatchSize, double learningRate, const double learningRateFactor, const int epochs, 
					      const int epochsFactorUpdate, int steps, const double stepsIncremantal, double momentum, const double momentumFactor, double lambda, 
					      const double lambdaFactor) {
	// Device memory allocation and copying	
	double* reconstructionErrors = new double[epochs];	
	double* devicePreviousDeltaWeights; gpuErrorCheckCuda(cudaMalloc(&devicePreviousDeltaWeights, sizeof(double)*numVisible*numHidden));
	double* deviceData; gpuErrorCheckCuda(cudaMalloc(&deviceData, sizeof(double)*numVisible*samples));
	gpuErrorCheckCuda(cudaMemcpy(deviceData, data, sizeof(double)*numVisible*samples, cudaMemcpyHostToDevice));
	
	// Define number of minibatches
	int numMiniBatches = static_cast<int>(ceil(1.0*samples/miniBatchSize));
	if (miniBatchSize > samples)
		miniBatchSize = samples;

	// Local parameters
   	double localLambda = -lambda;
   	double localMomentum = momentum;
	
   	for (int i = 0; i < epochs; i++) {
		// Changing parameters during training
		if(epochsFactorUpdate != 0 && i % epochsFactorUpdate == 0) {
			steps += stepsIncremantal;			
			if(learningRateFactor != 0)			
				learningRate /= learningRateFactor;
			if (lambdaFactor != 0)
				localLambda /= lambdaFactor;
			
			if (momentumFactor != 0)			
				localMomentum /= momentumFactor;			
		}

		for (int j = 0; j < numMiniBatches; j++) {
			// Caluclate current size of minibatch				
			int currTranslation = j*miniBatchSize;
			int nextTranslation = (j + 1)*miniBatchSize;
			int currMiniBatchSize = nextTranslation > samples ? samples - currTranslation : miniBatchSize;
			
			// Prepare mini batch
			double* deviceMiniBatch; gpuErrorCheckCuda(cudaMalloc(&deviceMiniBatch, sizeof(double)*numVisible*currMiniBatchSize));
			for(int k = 0; k < numVisible; k++)
				gpuErrorCheckCublas(cublasDcopy(defaultHandle, currMiniBatchSize, deviceData + j*currMiniBatchSize + k*samples , 1, deviceMiniBatch + k*currMiniBatchSize, 1));

			#ifdef DEBUG			
				Utility::printArrayGPU("MiniBatch", deviceMiniBatch, currMiniBatchSize*numVisible);
			#endif

			// Train a model with a mini batch
			Utility::Deltas d = TrainingIteration(deviceMiniBatch, currMiniBatchSize, learningRate, steps);
        		
		    	// Weights-decay
		    	gpuErrorCheckCublas(cublasDaxpy(defaultHandle, numHidden*numVisible, &localLambda, deviceWeights, 1, d.deviceDeltaWeights, 1));
	
		    	// Normal update
		    	const double alpha = 1.0;
		    	gpuErrorCheckCublas(cublasDaxpy(defaultHandle, numHidden*numVisible, &alpha, d.deviceDeltaWeights, 1, deviceWeights, 1));
		    	gpuErrorCheckCublas(cublasDaxpy(defaultHandle, numHidden, &alpha, d.deviceDeltaBiasesHidden, 1, deviceBiasesHidden, 1));
			gpuErrorCheckCublas(cublasDaxpy(defaultHandle, numVisible, &alpha, d.deviceDeltaBiasesVisible, 1, deviceBiasesVisible, 1));

		    	// Momentum update
		    	if (j > 0)
		    	    gpuErrorCheckCublas(cublasDaxpy(defaultHandle, numHidden*numVisible, &localMomentum, devicePreviousDeltaWeights, 1, deviceWeights, 1));
		    	gpuErrorCheckCublas(cublasDcopy(defaultHandle, numHidden*numVisible, d.deviceDeltaWeights, 1, devicePreviousDeltaWeights, 1));

			// Free minibatch memory
			gpuErrorCheckCuda(cudaFree(deviceMiniBatch));
		}
		

		// Reconstruction error (Gibbs chain = 1)
		Utility::Gradients g = ContrastiveDivergence(deviceData, samples, 1);
		const double minusOne = -1.0;
		gpuErrorCheckCublas(cublasDaxpy(defaultHandle, samples*numVisible, &minusOne, g.deviceVisiblePositiveGradients, 1, g.deviceVisibleNegativeGradients, 1));
		
		// Memory allocation
		thrust::device_ptr<double> deviceIt;	
		double* deviceOnes; gpuErrorCheckCuda(cudaMalloc(&deviceOnes, sizeof(double)*numVisible));
		deviceIt = thrust::device_pointer_cast(deviceOnes); thrust::fill_n(deviceIt, numVisible + 1, 1.0);
		double* deviceReconstructionErrors; gpuErrorCheckCuda(cudaMalloc(&deviceReconstructionErrors, sizeof(double)*samples));
		deviceIt = thrust::device_pointer_cast(deviceReconstructionErrors);	

		// Sum errors for each sample
		double alpha = 1.0;
		const double beta = 0.0;
		gpuErrorCheckCublas(cublasDgemv(defaultHandle, CUBLAS_OP_N, samples, numVisible, &alpha, g.deviceVisibleNegativeGradients, samples, deviceOnes, 1, &beta, deviceReconstructionErrors, 1));
		
		#ifdef DEBUG
			Utility::printArrayGPU("Reconstructions per samples", deviceReconstructionErrors, samples);
		#endif
		
		// Mean error from above sums
		double* deviceError; gpuErrorCheckCuda(cudaMalloc(&deviceError, sizeof(double)));
		alpha = 1.0 / samples;
		gpuErrorCheckCublas(cublasDgemv(defaultHandle, CUBLAS_OP_N, 1, samples, &alpha, deviceReconstructionErrors, 1, deviceReconstructionErrors, 1, &beta, deviceError, 1));
		gpuErrorCheckCuda(cudaMemcpy(reconstructionErrors + i, deviceError, sizeof(double), cudaMemcpyDeviceToHost));

		// Free memory
		gpuErrorCheckCuda(cudaFree(deviceError));
		gpuErrorCheckCuda(cudaFree(deviceReconstructionErrors));
		gpuErrorCheckCuda(cudaFree(deviceOnes));

		// Print report
		printf("Epochs %d/%d with mean reconstruction error: %.10f\n", i+1, epochs, reconstructionErrors[i]) ;
	}
    	
	Utility::printArrayCPU("Errors", reconstructionErrors, epochs);

	// Free memory
	delete[] reconstructionErrors;
    	gpuErrorCheckCuda(cudaFree(devicePreviousDeltaWeights));
	gpuErrorCheckCuda(cudaFree(deviceData));
	gpuErrorCheckCuda(cudaMemcpy(weights, deviceWeights, sizeof(double)*numVisible*numHidden, cudaMemcpyDeviceToHost));
	gpuErrorCheckCuda(cudaMemcpy(biasesHidden, deviceBiasesHidden, sizeof(double)*numHidden, cudaMemcpyDeviceToHost));
	gpuErrorCheckCuda(cudaMemcpy(biasesVisible, deviceBiasesVisible, sizeof(double)*numVisible, cudaMemcpyDeviceToHost));
}
