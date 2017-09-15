#ifndef RBM_H
#define RBM_H

#include <cublas_v2.h>
#include "Utility.cuh"

class RestrictedBoltzmannMachine {
	protected:
		double* deviceWeights;
		double* deviceBiasesVisible;
		double* deviceBiasesHidden;
		int numVisible;
		int numHidden;
		cublasHandle_t defaultHandle;

		virtual void HiddenTransformation(double* const deviceHiddenMatrix, const int samples) const = 0;
		virtual void VisibleTransformation(double* const deviceVisibleMatrix, const int samples) const = 0;	
		
		void GetVisibleGivenHidden(double* const deviceVisibleMatrix, const double* const deviceHiddenMatrix, const double* const deviceResizedBiasesVisible, const int samples) const;
		Utility::Gradients ContrastiveDivergence(double* const deviceData, const int samples, const int steps) const;
		Utility::Deltas GetDeltaWeights(Utility::Gradients& gradients, const int samples, const double learningRate) const;
		Utility::Deltas TrainingIteration(double* const deviceData, const int samples, const double learningRate, const int steps) const;

	public:
        	double* weights;
		double* biasesVisible;
		double* biasesHidden;

        	RestrictedBoltzmannMachine(const double* const weights_, const double* const biasesVisible_, const double* const biasesHidden_, const int numHidden_, const int numVisible_);
		virtual ~RestrictedBoltzmannMachine();
		void GetHiddenGivenVisible(double* const deviceHiddenMatrix, const double* const deviceVisibleMatrix, const double* const deviceResizedBiasesHidden, const int samples) const;	
		void TrainNetwork(const double* const data, const int samples, int miniBatchSize, double learningRate, const double learningRateFactor, const int epochs, const int epochsFactorUpdate, 
				  int steps, const double stepsIncremantal, double momentum, const double momentumFactor, double lambda, const double lambdaFactor);
		double* GibbsChainIteration(const double* const data, const int samples, int steps, const char* mode) const;
};

class BinaryBinaryRestrictedBoltzmannMachine : public RestrictedBoltzmannMachine {
	public:
		void HiddenTransformation(double* const deviceHiddenMatrix, const int samples) const;
		void VisibleTransformation(double* const deviceVisibleMatrix, const int samples) const;	
		
		BinaryBinaryRestrictedBoltzmannMachine(const double* const weights_, const double* const biasesVisible_, const double* const biasesHidden_, const int numHidden_, 
					  	       const int numVisible_);
		~BinaryBinaryRestrictedBoltzmannMachine();
};

class LinearNReluRestrictedBoltzmannMachine : public RestrictedBoltzmannMachine {
	public:
		void HiddenTransformation(double* const deviceHiddenMatrix, const int samples) const;
		void VisibleTransformation(double* const deviceVisibleMatrix, const int samples) const;	
		
		LinearNReluRestrictedBoltzmannMachine(const double* const weights_, const double* const biasesVisible_, const double* const biasesHidden_, const int numHidden_, const int numVisible_);
		~LinearNReluRestrictedBoltzmannMachine();
};

class LinearBinaryRestrictedBoltzmannMachine : public RestrictedBoltzmannMachine {
	public:
		void HiddenTransformation(double* const deviceHiddenMatrix, const int samples) const;
		void VisibleTransformation(double* const deviceVisibleMatrix, const int samples) const;	
		
		LinearBinaryRestrictedBoltzmannMachine(const double* const weights_, const double* const biasesVisible_, const double* const biasesHidden_, const int numHidden_,
						       const int numVisible_);
		~LinearBinaryRestrictedBoltzmannMachine();
};


#endif
