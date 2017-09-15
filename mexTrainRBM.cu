#include <cuda_runtime.h>
#include <string>
#include "mex.h"
#include "RestrictedBoltzmannMachine.cuh"
#include "Utility.cuh"


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{   
    
    std::string mode(mxArrayToString(prhs[4]));
    double* params;
    params = mxGetPr(prhs[5]); int hidden = params[0];
    params = mxGetPr(prhs[6]); int miniBatchSize = params[0];
    params = mxGetPr(prhs[7]); double learningRate = params[0];
    params = mxGetPr(prhs[8]); double learningRateFactor = params[0];
    params = mxGetPr(prhs[9]); int steps = params[0];
    params = mxGetPr(prhs[10]); int stepsIncremental = params[0];
    params = mxGetPr(prhs[11]); int epochs = params[0];
    params = mxGetPr(prhs[12]); int epochsFactorUpdate = params[0];
    params = mxGetPr(prhs[13]); double momentum = params[0];
    params = mxGetPr(prhs[14]); double momentumFactor = params[0];
    params = mxGetPr(prhs[15]); double lambda = params[0];
    params = mxGetPr(prhs[16]); double lambdaFactor = params[0];

    
    double* data = mxGetPr(prhs[0]);
    int visible = mxGetN(prhs[0]);
    int samples = mxGetM(prhs[0]);
    
    double* weights = mxGetPr(prhs[1]);
    double* biasesVisible = mxGetPr(prhs[2]);
    double* biasesHidden = mxGetPr(prhs[3]);

    RestrictedBoltzmannMachine* rbm;
    if (mode == "LinearNRelu") 
    	rbm = new LinearNReluRestrictedBoltzmannMachine(weights, biasesVisible, biasesHidden, hidden, visible);    
    else if (mode == "BinaryBinary")    
	rbm = new BinaryBinaryRestrictedBoltzmannMachine(weights, biasesVisible, biasesHidden, hidden, visible);
    else if (mode == "LinearBinary")    
	rbm = new LinearBinaryRestrictedBoltzmannMachine(weights, biasesVisible, biasesHidden, hidden, visible);
    else 
	mexErrMsgTxt("Incorrect mode parameters. You can pass the one from the following set {BinaryBinary, LinearBinary, LinearNRelu}"); 
  
    rbm->TrainNetwork(data, samples, miniBatchSize, learningRate, learningRateFactor, epochs, epochsFactorUpdate, steps, stepsIncremental, momentum, momentumFactor, lambda, lambdaFactor);
    
    plhs[0] = mxCreateDoubleMatrix(hidden, visible, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(visible, 1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(hidden, 1, mxREAL);

    double* weightsOutput = mxGetPr(plhs[0]);
    double* biasesVisibleOutput = mxGetPr(plhs[1]);
    double* biasesHiddenOutput = mxGetPr(plhs[2]);

    for (int i = 0; i < visible*hidden; i++) {
        weightsOutput[i] = rbm->weights[i];
        if (i < hidden)
            biasesHiddenOutput[i] = rbm->biasesHidden[i];
        if (i < visible)
            biasesVisibleOutput[i] = rbm->biasesVisible[i];
    }

    delete rbm;
    cudaDeviceReset();
}
