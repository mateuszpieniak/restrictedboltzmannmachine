#include <iostream>
#include <ctime>
#include <cuda_runtime.h>
#include <string>
#include <thrust/device_ptr.h>

#include "mex.h"
#include "gpu/mxGPUArray.h"

#include "RestrictedBoltzmannMachine.cuh"
#include "Utility.cuh"


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{   
    
    std::string modeType(mxArrayToString(prhs[4]));
    std::string modeWork(mxArrayToString(prhs[5]));
    double* params;
    params = mxGetPr(prhs[6]); int hidden = params[0];
    params = mxGetPr(prhs[7]); int steps = params[0];
    
    double* data = mxGetPr(prhs[0]);
    int visible = mxGetN(prhs[0]);
    int samples = mxGetM(prhs[0]);
    
    double* weights = mxGetPr(prhs[1]);
    double* biasesVisible = mxGetPr(prhs[2]);
    double* biasesHidden = mxGetPr(prhs[3]);

    RestrictedBoltzmannMachine* rbm;
    if (modeType == "LinearNRelu") 
    	rbm = new LinearNReluRestrictedBoltzmannMachine(weights, biasesVisible, biasesHidden, hidden, visible);    
    else if (modeType == "BinaryBinary")    
	rbm = new BinaryBinaryRestrictedBoltzmannMachine(weights, biasesVisible, biasesHidden, hidden, visible);
    else if (modeType == "LinearBinary")    
	rbm = new LinearBinaryRestrictedBoltzmannMachine(weights, biasesVisible, biasesHidden, hidden, visible);
    else 
	mexErrMsgTxt("Incorrect mode parameters. You can pass the one from the following set {BinaryBinary, LinearBinary, LinearNRelu}"); 

    double* result;
    int n; 
    if (modeWork == "Sample") {
    	 result = rbm->GibbsChainIteration(data, samples, steps, "Sample");
 	 n = visible; 
    }
    else if (modeWork == "Reconstruction") {    
	 result = rbm->GibbsChainIteration(data, samples, 1, "Reconstruction");
	 n = hidden;
    }
    else 
	mexErrMsgTxt("Incorrect mode parameters. You can pass the one from the following set {Sample, Reconstruction}"); 
  
    plhs[0] = mxCreateDoubleMatrix(samples, n, mxREAL);
    double* resultOutput = mxGetPr(plhs[0]);

    for (int i = 0; i < samples*n; i++)
        resultOutput[i] = result[i];

    delete rbm;
    cudaDeviceReset();
}
