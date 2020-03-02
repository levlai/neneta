#include <gtest/gtest.h>
#include <ConvLayer.h>
#include <OpenCLContext.h>
#include <OpenCLProgram.h>
#include <OpenCLExecutionPlan.h>


using namespace neneta;
using namespace neneta::net;


extern conf::ConfigurationReader envReader;
void printComplexMatrix(cmn::GPUFLOAT reCalculatedResult[], cmn::GPUFLOAT imCalculatedResult[], int width);

TEST(ConvLayerBpMCTests, test_multichannel_kernel)
{
    const int numOfKernels = 1;
    const int numOfChannels = 2;

    gpu::OpenCLContext oclContext(envReader);
    oclContext.printInfo();
    gpu::OpenCLProgram oclProgram(envReader);

    std::vector<cmn::GPUFLOAT> tmp_weights = {1,2,3, //repart
                                     4,5,6,
                                     7,8,9,
                                     1,2,3, //impart
                                     4,5,6,
                                     7,8,9};
    std::vector<cmn::GPUFLOAT> tmp_reInput = {0.2858, 0.5308, 0.3371, 0.2630, 0.9133, 0.1067, 0.3998,
                                  0.7572, 0.7792, 0.1622, 0.6541, 0.1524, 0.9619, 0.2599,
                                  0.7537, 0.9340, 0.7943, 0.6892, 0.8258, 0.0046, 0.8001,
                                  0.3804, 0.1299, 0.3112, 0.7482, 0.5383, 0.7749, 0.4314,
                                  0.5678, 0.5688, 0.5285, 0.4505, 0.9961, 0.8173, 0.9106,
                                  0.0759, 0.4694, 0.1656, 0.0838, 0.0782, 0.8687, 0.1818,
                                  0.0540, 0.0119, 0.6020, 0.2290, 0.4427, 0.0844, 0.2638};
    std::vector<cmn::GPUFLOAT> tmp_imInput = {0.1455, 0.6221, 0.1839, 0.4893, 0.2417, 0.0598, 0.6491,
                                  0.1361, 0.3510, 0.2400, 0.3377, 0.4039, 0.2348, 0.7317,
                                  0.8693, 0.5132, 0.4173, 0.9001, 0.0965, 0.3532, 0.6477,
                                  0.5797, 0.4018, 0.0497, 0.3692, 0.1320, 0.8212, 0.4509,
                                  0.5499, 0.0760, 0.9027, 0.1112, 0.9421, 0.0154, 0.5470,
                                  0.1450, 0.2399, 0.9448, 0.7803, 0.9561, 0.0430, 0.2963,
                                  0.8530, 0.1233, 0.4909, 0.3897, 0.5752, 0.1690, 0.7447};
  /* With "fake" */
    std::vector<cmn::GPUFLOAT> tmp_reReInput = { 0.8147, 0.0975, 0.1576, 0.1419, 0.6557,
                                   0.9058, 0.2785, 0.9706, 0.4218, 0.0357,
                                   0.1270, 0.5469, 0.9572, 0.9157, 0.8491,
                                   0.9134, 0.9575, 0.4854, 0.7922, 0.9340,
                                   0.6324, 0.9649, 0.8003, 0.9595, 0.6787};
    std::vector<cmn::GPUFLOAT> tmp_imReInput = { 0.7577, 0.7060, 0.8235, 0.4387, 0.4898,
                                   0.7431, 0.0318, 0.6948, 0.3816, 0.4456,
                                   0.3922, 0.2769, 0.3171, 0.7655, 0.6463,
                                   0.6555, 0.0462, 0.9502, 0.7952, 0.7094,
                                   0.1712, 0.0971, 0.0344, 0.1869, 0.7547};


    std::vector<cmn::GPUFLOAT> tmp_reResult = {numOfKernels*13.6395 ,  numOfKernels*24.8329 ,  numOfKernels*36.7274 ,  numOfKernels*30.9574,   numOfKernels*27.3638,   numOfKernels*13.9785,  numOfKernels*  6.8565,
                                       numOfKernels*22.4688 ,  numOfKernels*28.5121 ,  numOfKernels*46.4225 ,  numOfKernels*37.5778,   numOfKernels*40.7049,   numOfKernels*20.8788,  numOfKernels* 10.1564,
                                       numOfKernels*20.5233 ,  numOfKernels*26.3393 ,  numOfKernels*39.5098 ,  numOfKernels*41.8495,   numOfKernels*49.3831,   numOfKernels*31.4034,  numOfKernels* 13.5925,
                                       numOfKernels*20.9640 ,  numOfKernels*21.7276 ,  numOfKernels*42.5138 ,  numOfKernels*49.8685,   numOfKernels*62.3325,   numOfKernels*37.6161,  numOfKernels* 15.9932,
                                       numOfKernels*13.2999 ,  numOfKernels*14.8258 ,  numOfKernels*25.4728 ,  numOfKernels*31.1043,   numOfKernels*48.6979,   numOfKernels*32.2637,  numOfKernels* 17.5336,
                                       numOfKernels* 5.9868 ,  numOfKernels* 5.7759 ,  numOfKernels* 9.9506 ,  numOfKernels*12.0285,   numOfKernels*20.5386,   numOfKernels*13.4702,  numOfKernels*  7.4564,
                                       numOfKernels* 1.0269 ,  numOfKernels* 1.2672 ,  numOfKernels* 0.9374 ,  numOfKernels* 1.4534,   numOfKernels* 5.3447,   numOfKernels* 3.3926,  numOfKernels*  1.5094};
    std::vector<cmn::GPUFLOAT> tmp_imResult = {numOfKernels*-14.6655 , numOfKernels*-14.7919 , numOfKernels*-15.8050,  numOfKernels* -6.4414,  numOfKernels*-16.2802 , numOfKernels*-12.4779,  numOfKernels* -9.1805,
                                       numOfKernels*-26.0814 , numOfKernels*-28.8239 , numOfKernels*-43.9933,  numOfKernels*-31.0792,  numOfKernels*-31.5277 , numOfKernels*-14.1680,  numOfKernels* -5.7458,
                                       numOfKernels*-18.0441 , numOfKernels*-28.1205 , numOfKernels*-52.4022,  numOfKernels*-58.1271,  numOfKernels*-60.5645 , numOfKernels*-33.0438,  numOfKernels*-13.4845,
                                       numOfKernels*-23.4000 , numOfKernels*-44.9764 , numOfKernels*-63.5654,  numOfKernels*-67.3371,  numOfKernels*-67.1295 , numOfKernels*-42.8381,  numOfKernels*-19.9402,
                                       numOfKernels*-23.1051 , numOfKernels*-51.8992 , numOfKernels*-69.5876,  numOfKernels*-76.0211,  numOfKernels*-72.4573 , numOfKernels*-45.1977,  numOfKernels*-18.6720,
                                       numOfKernels*-13.0686 , numOfKernels*-27.3009 , numOfKernels*-32.8800,  numOfKernels*-35.8455,  numOfKernels*-33.8850 , numOfKernels*-19.7834,  numOfKernels* -7.2976,
                                       numOfKernels* -3.7941 , numOfKernels* -8.3188 , numOfKernels* -9.9258,  numOfKernels*-10.8878,  numOfKernels* -9.5107 , numOfKernels* -4.6338,  numOfKernels* -1.3574};

    std::vector<cmn::GPUFLOAT> tmp_reNewW = {12.4344,   14.2176,   13.2687,
                                 18.6388,   20.3466,   19.8105,
                                 21.0369,   24.4087,   23.4088};
    std::vector<cmn::GPUFLOAT> tmp_imNewW = {-15.6704,  -11.7766,  -14.5057,
                                 -11.3171,   -8.0477,  -10.3620,
                                  -6.6281,   -4.6506,   -4.8763};


    std::vector<cmn::GPUFLOAT> tmp_bias = {1,1};
    std::vector<cmn::GPUFLOAT> bias;
    std::vector<cmn::GPUFLOAT> weights;
    std::vector<cmn::GPUFLOAT> reResult;
    std::vector<cmn::GPUFLOAT> imResult;
    std::vector<cmn::GPUFLOAT> reInput;
    std::vector<cmn::GPUFLOAT> imInput;
    std::vector<cmn::GPUFLOAT> reNewW;
    std::vector<cmn::GPUFLOAT> imNewW;
    std::vector<cmn::GPUFLOAT> reReInput;
    std::vector<cmn::GPUFLOAT> imReInput;
    for(int i = 0; i < numOfKernels; ++i)
    {
        for(int i = 0; i < numOfChannels; ++i)
        {
            weights.insert(weights.end(), tmp_weights.begin(), tmp_weights.end());
            reNewW.insert(reNewW.end(), tmp_reNewW.begin(), tmp_reNewW.end());
            imNewW.insert(imNewW.end(), tmp_imNewW.begin(), tmp_imNewW.end());
        }
        bias.insert(bias.end(), tmp_bias.begin(), tmp_bias.end());
        reReInput.insert(reReInput.end(), tmp_reReInput.begin(), tmp_reReInput.end());
        imReInput.insert(imReInput.end(), tmp_imReInput.begin(), tmp_imReInput.end());
    }
    for(int i = 0; i < numOfChannels; ++i)
    {
        reResult.insert(reResult.end(), tmp_reResult.begin(), tmp_reResult.end());
        imResult.insert(imResult.end(), tmp_imResult.begin(), tmp_imResult.end());
        reInput.insert(reInput.end(), tmp_reInput.begin(), tmp_reInput.end());
        imInput.insert(imInput.end(), tmp_imInput.begin(), tmp_imInput.end());
    }
    const int inputSize  = 7;
    const int stride = 1;
    const int kernelSize = 3;
    conf::ConvLayerParams clayerParams(numOfChannels/*channels*/, numOfKernels/*kernels*/, kernelSize/*kernelsize*/, stride/*stride*/,
                                       2/*inputDim*/, inputSize/*inputSize*/, 0.1/*weightsdev*/, 0, conf::ConvLayerParams::WeightsType::COMPLEX, bias.front(), bias.back(), "fake"/*actFunc*/, "convtest"/*id*/);

    if(oclProgram.compile(oclContext))
    {
        //storage for result
        const unsigned int resSize = inputSize*inputSize;
        cmn::GPUFLOAT reCalculatedResult[numOfChannels*resSize];
        cmn::GPUFLOAT imCalculatedResult[numOfChannels*resSize];

        net::ConvLayer uut(clayerParams, weights, bias, envReader, oclProgram, oclContext);

        gpu::BufferIO input;
        //allocate enough memory on gpu (10 x input size should be enough)
        input.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, std::max(numOfChannels, numOfKernels)*sizeof(reCalculatedResult));
        input.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, std::max(numOfChannels, numOfKernels)*sizeof(imCalculatedResult));
        input.m_reShMemBkp = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, std::max(numOfChannels, numOfKernels)*sizeof(reCalculatedResult));
        input.m_imShMemBkp = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, std::max(numOfChannels, numOfKernels)*sizeof(imCalculatedResult));
        uut.writeToBuffer(oclContext.getCommandQueue(), input.m_reShMem, reInput.size()*sizeof(cmn::GPUFLOAT), reInput.data());
        uut.writeToBuffer(oclContext.getCommandQueue(), input.m_imShMem, imInput.size()*sizeof(cmn::GPUFLOAT), imInput.data());
        uut.setInput(input);
        uut.runFwdPropagation(oclContext);
        uut.printFwdProfilingInfo();

        //back propagation
        gpu::BufferIO bkpInput = uut.getOutput();
        uut.writeToBuffer(oclContext.getCommandQueue(), bkpInput.m_reShMem, reReInput.size()*sizeof(cmn::GPUFLOAT), reReInput.data());
        uut.writeToBuffer(oclContext.getCommandQueue(), bkpInput.m_imShMem, imReInput.size()*sizeof(cmn::GPUFLOAT), imReInput.data());

        uut.setBkpInput(bkpInput);
        uut.runBckPropagation(oclContext);
        uut.printBckProfilingInfo();

        LayerWeights newWeights = uut.getWeights();
        cmn::GPUFLOAT reNewWeights[numOfChannels*kernelSize*kernelSize];
        cmn::GPUFLOAT imNewWeights[numOfChannels*kernelSize*kernelSize];

        //check result for each kernel
        for(int kernelId = 0; kernelId < numOfKernels; ++kernelId)
        {
            uut.readFromBuffer(oclContext.getCommandQueue(), newWeights[kernelId].m_re, sizeof(reNewWeights), reNewWeights);
            uut.readFromBuffer(oclContext.getCommandQueue(), newWeights[kernelId].m_im, sizeof(imNewWeights), imNewWeights);

            for(unsigned int channelId = 0; channelId < numOfChannels; ++channelId)
            {
                for(int k = 0; k < kernelSize*kernelSize; ++k)
                {
                    unsigned int kdx = kernelSize*kernelSize*channelId + k;
                    EXPECT_NEAR(reNewW[kernelId*kernelSize*kernelSize*numOfChannels + kdx], reNewWeights[kdx], 0.01);
                    EXPECT_NEAR(imNewW[kernelId*kernelSize*kernelSize*numOfChannels + kdx], imNewWeights[kdx], 0.01);
                }
            }
        }
        //check propagated error
        gpu::BufferIO output = uut.getOutput();
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_reShMem, sizeof(reCalculatedResult), reCalculatedResult);
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_imShMem, sizeof(imCalculatedResult), imCalculatedResult);
        for(unsigned int channelId = 0; channelId < numOfChannels; ++channelId)
        {
            unsigned int idx = resSize*channelId;
            for(size_t i = 0; i < resSize; ++i)
            {
                EXPECT_NEAR(reResult[idx + i], reCalculatedResult[idx + i], 0.01);
                EXPECT_NEAR(imResult[idx + i], imCalculatedResult[idx + i], 0.01);
            }
        }
    }
}

TEST(ConvLayerBpMCTests, test_channel_multikernel)
{
    const int numOfKernels = 2;
    const int numOfChannels = 1;

    gpu::OpenCLContext oclContext(envReader);
    oclContext.printInfo();
    gpu::OpenCLProgram oclProgram(envReader);

    std::vector<cmn::GPUFLOAT> tmp_weights = {1,2,3, //repart
                                     4,5,6,
                                     7,8,9,
                                     1,2,3, //impart
                                     4,5,6,
                                     7,8,9};
    std::vector<cmn::GPUFLOAT> tmp_reInput = {0.2858, 0.5308, 0.3371, 0.2630, 0.9133, 0.1067, 0.3998,
                                  0.7572, 0.7792, 0.1622, 0.6541, 0.1524, 0.9619, 0.2599,
                                  0.7537, 0.9340, 0.7943, 0.6892, 0.8258, 0.0046, 0.8001,
                                  0.3804, 0.1299, 0.3112, 0.7482, 0.5383, 0.7749, 0.4314,
                                  0.5678, 0.5688, 0.5285, 0.4505, 0.9961, 0.8173, 0.9106,
                                  0.0759, 0.4694, 0.1656, 0.0838, 0.0782, 0.8687, 0.1818,
                                  0.0540, 0.0119, 0.6020, 0.2290, 0.4427, 0.0844, 0.2638};
    std::vector<cmn::GPUFLOAT> tmp_imInput = {0.1455, 0.6221, 0.1839, 0.4893, 0.2417, 0.0598, 0.6491,
                                  0.1361, 0.3510, 0.2400, 0.3377, 0.4039, 0.2348, 0.7317,
                                  0.8693, 0.5132, 0.4173, 0.9001, 0.0965, 0.3532, 0.6477,
                                  0.5797, 0.4018, 0.0497, 0.3692, 0.1320, 0.8212, 0.4509,
                                  0.5499, 0.0760, 0.9027, 0.1112, 0.9421, 0.0154, 0.5470,
                                  0.1450, 0.2399, 0.9448, 0.7803, 0.9561, 0.0430, 0.2963,
                                  0.8530, 0.1233, 0.4909, 0.3897, 0.5752, 0.1690, 0.7447};
  /* With "fake" */
    std::vector<cmn::GPUFLOAT> tmp_reReInput = { 0.8147, 0.0975, 0.1576, 0.1419, 0.6557,
                                   0.9058, 0.2785, 0.9706, 0.4218, 0.0357,
                                   0.1270, 0.5469, 0.9572, 0.9157, 0.8491,
                                   0.9134, 0.9575, 0.4854, 0.7922, 0.9340,
                                   0.6324, 0.9649, 0.8003, 0.9595, 0.6787};
    std::vector<cmn::GPUFLOAT> tmp_imReInput = { 0.7577, 0.7060, 0.8235, 0.4387, 0.4898,
                                   0.7431, 0.0318, 0.6948, 0.3816, 0.4456,
                                   0.3922, 0.2769, 0.3171, 0.7655, 0.6463,
                                   0.6555, 0.0462, 0.9502, 0.7952, 0.7094,
                                   0.1712, 0.0971, 0.0344, 0.1869, 0.7547};


    std::vector<cmn::GPUFLOAT> tmp_reResult = {numOfKernels*13.6395 ,  numOfKernels*24.8329 ,  numOfKernels*36.7274 ,  numOfKernels*30.9574,   numOfKernels*27.3638,   numOfKernels*13.9785,  numOfKernels*  6.8565,
                                       numOfKernels*22.4688 ,  numOfKernels*28.5121 ,  numOfKernels*46.4225 ,  numOfKernels*37.5778,   numOfKernels*40.7049,   numOfKernels*20.8788,  numOfKernels* 10.1564,
                                       numOfKernels*20.5233 ,  numOfKernels*26.3393 ,  numOfKernels*39.5098 ,  numOfKernels*41.8495,   numOfKernels*49.3831,   numOfKernels*31.4034,  numOfKernels* 13.5925,
                                       numOfKernels*20.9640 ,  numOfKernels*21.7276 ,  numOfKernels*42.5138 ,  numOfKernels*49.8685,   numOfKernels*62.3325,   numOfKernels*37.6161,  numOfKernels* 15.9932,
                                       numOfKernels*13.2999 ,  numOfKernels*14.8258 ,  numOfKernels*25.4728 ,  numOfKernels*31.1043,   numOfKernels*48.6979,   numOfKernels*32.2637,  numOfKernels* 17.5336,
                                       numOfKernels* 5.9868 ,  numOfKernels* 5.7759 ,  numOfKernels* 9.9506 ,  numOfKernels*12.0285,   numOfKernels*20.5386,   numOfKernels*13.4702,  numOfKernels*  7.4564,
                                       numOfKernels* 1.0269 ,  numOfKernels* 1.2672 ,  numOfKernels* 0.9374 ,  numOfKernels* 1.4534,   numOfKernels* 5.3447,   numOfKernels* 3.3926,  numOfKernels*  1.5094};
    std::vector<cmn::GPUFLOAT> tmp_imResult = {numOfKernels*-14.6655 , numOfKernels*-14.7919 , numOfKernels*-15.8050,  numOfKernels* -6.4414,  numOfKernels*-16.2802 , numOfKernels*-12.4779,  numOfKernels* -9.1805,
                                       numOfKernels*-26.0814 , numOfKernels*-28.8239 , numOfKernels*-43.9933,  numOfKernels*-31.0792,  numOfKernels*-31.5277 , numOfKernels*-14.1680,  numOfKernels* -5.7458,
                                       numOfKernels*-18.0441 , numOfKernels*-28.1205 , numOfKernels*-52.4022,  numOfKernels*-58.1271,  numOfKernels*-60.5645 , numOfKernels*-33.0438,  numOfKernels*-13.4845,
                                       numOfKernels*-23.4000 , numOfKernels*-44.9764 , numOfKernels*-63.5654,  numOfKernels*-67.3371,  numOfKernels*-67.1295 , numOfKernels*-42.8381,  numOfKernels*-19.9402,
                                       numOfKernels*-23.1051 , numOfKernels*-51.8992 , numOfKernels*-69.5876,  numOfKernels*-76.0211,  numOfKernels*-72.4573 , numOfKernels*-45.1977,  numOfKernels*-18.6720,
                                       numOfKernels*-13.0686 , numOfKernels*-27.3009 , numOfKernels*-32.8800,  numOfKernels*-35.8455,  numOfKernels*-33.8850 , numOfKernels*-19.7834,  numOfKernels* -7.2976,
                                       numOfKernels* -3.7941 , numOfKernels* -8.3188 , numOfKernels* -9.9258,  numOfKernels*-10.8878,  numOfKernels* -9.5107 , numOfKernels* -4.6338,  numOfKernels* -1.3574};

    std::vector<cmn::GPUFLOAT> tmp_reNewW = {12.4344,   14.2176,   13.2687,
                                 18.6388,   20.3466,   19.8105,
                                 21.0369,   24.4087,   23.4088};
    std::vector<cmn::GPUFLOAT> tmp_imNewW = {-15.6704,  -11.7766,  -14.5057,
                                 -11.3171,   -8.0477,  -10.3620,
                                  -6.6281,   -4.6506,   -4.8763};


    std::vector<cmn::GPUFLOAT> tmp_bias = {1,1};
    std::vector<cmn::GPUFLOAT> bias;
    std::vector<cmn::GPUFLOAT> weights;
    std::vector<cmn::GPUFLOAT> reResult;
    std::vector<cmn::GPUFLOAT> imResult;
    std::vector<cmn::GPUFLOAT> reInput;
    std::vector<cmn::GPUFLOAT> imInput;
    std::vector<cmn::GPUFLOAT> reNewW;
    std::vector<cmn::GPUFLOAT> imNewW;
    std::vector<cmn::GPUFLOAT> reReInput;
    std::vector<cmn::GPUFLOAT> imReInput;
    for(int i = 0; i < numOfKernels; ++i)
    {
        for(int i = 0; i < numOfChannels; ++i)
        {
            weights.insert(weights.end(), tmp_weights.begin(), tmp_weights.end());
            reNewW.insert(reNewW.end(), tmp_reNewW.begin(), tmp_reNewW.end());
            imNewW.insert(imNewW.end(), tmp_imNewW.begin(), tmp_imNewW.end());
        }
        bias.insert(bias.end(), tmp_bias.begin(), tmp_bias.end());
        reReInput.insert(reReInput.end(), tmp_reReInput.begin(), tmp_reReInput.end());
        imReInput.insert(imReInput.end(), tmp_imReInput.begin(), tmp_imReInput.end());
    }
    for(int i = 0; i < numOfChannels; ++i)
    {
        reResult.insert(reResult.end(), tmp_reResult.begin(), tmp_reResult.end());
        imResult.insert(imResult.end(), tmp_imResult.begin(), tmp_imResult.end());
        reInput.insert(reInput.end(), tmp_reInput.begin(), tmp_reInput.end());
        imInput.insert(imInput.end(), tmp_imInput.begin(), tmp_imInput.end());
    }
    const int inputSize  = 7;
    const int stride = 1;
    const int kernelSize = 3;
    conf::ConvLayerParams clayerParams(numOfChannels/*channels*/, numOfKernels/*kernels*/, kernelSize/*kernelsize*/, stride/*stride*/,
                                       2/*inputDim*/, inputSize/*inputSize*/, 0.1/*weightsdev*/, 0, conf::ConvLayerParams::WeightsType::COMPLEX, bias.front(), bias.back(), "fake"/*actFunc*/, "convtest"/*id*/);

    if(oclProgram.compile(oclContext))
    {
        //storage for result
        const unsigned int resSize = inputSize*inputSize;
        cmn::GPUFLOAT reCalculatedResult[numOfChannels*resSize];
        cmn::GPUFLOAT imCalculatedResult[numOfChannels*resSize];

        net::ConvLayer uut(clayerParams, weights, bias, envReader, oclProgram, oclContext);

        gpu::BufferIO input;
        //allocate enough memory on gpu (10 x input size should be enough)
        input.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, std::max(numOfChannels, numOfKernels)*sizeof(reCalculatedResult));
        input.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, std::max(numOfChannels, numOfKernels)*sizeof(imCalculatedResult));
        input.m_reShMemBkp = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, std::max(numOfChannels, numOfKernels)*sizeof(reCalculatedResult));
        input.m_imShMemBkp = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, std::max(numOfChannels, numOfKernels)*sizeof(imCalculatedResult));
        uut.writeToBuffer(oclContext.getCommandQueue(), input.m_reShMem, reInput.size()*sizeof(cmn::GPUFLOAT), reInput.data());
        uut.writeToBuffer(oclContext.getCommandQueue(), input.m_imShMem, imInput.size()*sizeof(cmn::GPUFLOAT), imInput.data());
        uut.setInput(input);
        uut.runFwdPropagation(oclContext);
        uut.printFwdProfilingInfo();

        //back propagation
        gpu::BufferIO bkpInput = uut.getOutput();
        uut.writeToBuffer(oclContext.getCommandQueue(), bkpInput.m_reShMem, reReInput.size()*sizeof(cmn::GPUFLOAT), reReInput.data());
        uut.writeToBuffer(oclContext.getCommandQueue(), bkpInput.m_imShMem, imReInput.size()*sizeof(cmn::GPUFLOAT), imReInput.data());

        uut.setBkpInput(bkpInput);
        uut.runBckPropagation(oclContext);
        uut.printBckProfilingInfo();

        gpu::BufferIO output = uut.getOutput();
        LayerWeights newWeights = uut.getWeights();
        cmn::GPUFLOAT reNewWeights[numOfChannels*kernelSize*kernelSize];
        cmn::GPUFLOAT imNewWeights[numOfChannels*kernelSize*kernelSize];
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_reShMem, sizeof(reCalculatedResult), reCalculatedResult);
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_imShMem, sizeof(imCalculatedResult), imCalculatedResult);
        //check result for each kernel
        for(int kernelId = 0; kernelId < numOfKernels; ++kernelId)
        {
            uut.readFromBuffer(oclContext.getCommandQueue(), newWeights[kernelId].m_re, sizeof(reNewWeights), reNewWeights);
            uut.readFromBuffer(oclContext.getCommandQueue(), newWeights[kernelId].m_im, sizeof(imNewWeights), imNewWeights);

            for(unsigned int channelId = 0; channelId < numOfChannels; ++channelId)
            {
                for(int k = 0; k < kernelSize*kernelSize; ++k)
                {
                    unsigned int kdx = kernelSize*kernelSize*channelId + k;
                    EXPECT_NEAR(reNewW[kernelId*kernelSize*kernelSize*numOfChannels + kdx], reNewWeights[kdx], 0.01);
                    EXPECT_NEAR(imNewW[kernelId*kernelSize*kernelSize*numOfChannels + kdx], imNewWeights[kdx], 0.01);
                }
            }
        }
        //check propagated error
        for(unsigned int channelId = 0; channelId < numOfChannels; ++channelId)
        {
            unsigned int idx = resSize*channelId;
            for(size_t i = 0; i < resSize; ++i)
            {
                EXPECT_NEAR(reResult[idx + i], reCalculatedResult[idx + i], 0.01);
                EXPECT_NEAR(imResult[idx + i], imCalculatedResult[idx + i], 0.01);
            }
        }
    }
}



TEST(ConvLayerBpMCTests, test_multichannel_multikernel)
{
    const int numOfKernels = 200;
    const int numOfChannels = 20;

    gpu::OpenCLContext oclContext(envReader);
    oclContext.printInfo();
    gpu::OpenCLProgram oclProgram(envReader);

    std::vector<cmn::GPUFLOAT> tmp_weights = {1,2,3, //repart
                                     4,5,6,
                                     7,8,9,
                                     1,2,3, //impart
                                     4,5,6,
                                     7,8,9};
    std::vector<cmn::GPUFLOAT> tmp_reInput = {0.2858, 0.5308, 0.3371, 0.2630, 0.9133, 0.1067, 0.3998,
                                  0.7572, 0.7792, 0.1622, 0.6541, 0.1524, 0.9619, 0.2599,
                                  0.7537, 0.9340, 0.7943, 0.6892, 0.8258, 0.0046, 0.8001,
                                  0.3804, 0.1299, 0.3112, 0.7482, 0.5383, 0.7749, 0.4314,
                                  0.5678, 0.5688, 0.5285, 0.4505, 0.9961, 0.8173, 0.9106,
                                  0.0759, 0.4694, 0.1656, 0.0838, 0.0782, 0.8687, 0.1818,
                                  0.0540, 0.0119, 0.6020, 0.2290, 0.4427, 0.0844, 0.2638};
    std::vector<cmn::GPUFLOAT> tmp_imInput = {0.1455, 0.6221, 0.1839, 0.4893, 0.2417, 0.0598, 0.6491,
                                  0.1361, 0.3510, 0.2400, 0.3377, 0.4039, 0.2348, 0.7317,
                                  0.8693, 0.5132, 0.4173, 0.9001, 0.0965, 0.3532, 0.6477,
                                  0.5797, 0.4018, 0.0497, 0.3692, 0.1320, 0.8212, 0.4509,
                                  0.5499, 0.0760, 0.9027, 0.1112, 0.9421, 0.0154, 0.5470,
                                  0.1450, 0.2399, 0.9448, 0.7803, 0.9561, 0.0430, 0.2963,
                                  0.8530, 0.1233, 0.4909, 0.3897, 0.5752, 0.1690, 0.7447};
  /* With "fake" */
    std::vector<cmn::GPUFLOAT> tmp_reReInput = { 0.8147, 0.0975, 0.1576, 0.1419, 0.6557,
                                   0.9058, 0.2785, 0.9706, 0.4218, 0.0357,
                                   0.1270, 0.5469, 0.9572, 0.9157, 0.8491,
                                   0.9134, 0.9575, 0.4854, 0.7922, 0.9340,
                                   0.6324, 0.9649, 0.8003, 0.9595, 0.6787};
    std::vector<cmn::GPUFLOAT> tmp_imReInput = { 0.7577, 0.7060, 0.8235, 0.4387, 0.4898,
                                   0.7431, 0.0318, 0.6948, 0.3816, 0.4456,
                                   0.3922, 0.2769, 0.3171, 0.7655, 0.6463,
                                   0.6555, 0.0462, 0.9502, 0.7952, 0.7094,
                                   0.1712, 0.0971, 0.0344, 0.1869, 0.7547};


    std::vector<cmn::GPUFLOAT> tmp_reResult = {numOfKernels*13.6395 ,  numOfKernels*24.8329 ,  numOfKernels*36.7274 ,  numOfKernels*30.9574,   numOfKernels*27.3638,   numOfKernels*13.9785,  numOfKernels*  6.8565,
                                       numOfKernels*22.4688 ,  numOfKernels*28.5121 ,  numOfKernels*46.4225 ,  numOfKernels*37.5778,   numOfKernels*40.7049,   numOfKernels*20.8788,  numOfKernels* 10.1564,
                                       numOfKernels*20.5233 ,  numOfKernels*26.3393 ,  numOfKernels*39.5098 ,  numOfKernels*41.8495,   numOfKernels*49.3831,   numOfKernels*31.4034,  numOfKernels* 13.5925,
                                       numOfKernels*20.9640 ,  numOfKernels*21.7276 ,  numOfKernels*42.5138 ,  numOfKernels*49.8685,   numOfKernels*62.3325,   numOfKernels*37.6161,  numOfKernels* 15.9932,
                                       numOfKernels*13.2999 ,  numOfKernels*14.8258 ,  numOfKernels*25.4728 ,  numOfKernels*31.1043,   numOfKernels*48.6979,   numOfKernels*32.2637,  numOfKernels* 17.5336,
                                       numOfKernels* 5.9868 ,  numOfKernels* 5.7759 ,  numOfKernels* 9.9506 ,  numOfKernels*12.0285,   numOfKernels*20.5386,   numOfKernels*13.4702,  numOfKernels*  7.4564,
                                       numOfKernels* 1.0269 ,  numOfKernels* 1.2672 ,  numOfKernels* 0.9374 ,  numOfKernels* 1.4534,   numOfKernels* 5.3447,   numOfKernels* 3.3926,  numOfKernels*  1.5094};
    std::vector<cmn::GPUFLOAT> tmp_imResult = {numOfKernels*-14.6655 , numOfKernels*-14.7919 , numOfKernels*-15.8050,  numOfKernels* -6.4414,  numOfKernels*-16.2802 , numOfKernels*-12.4779,  numOfKernels* -9.1805,
                                       numOfKernels*-26.0814 , numOfKernels*-28.8239 , numOfKernels*-43.9933,  numOfKernels*-31.0792,  numOfKernels*-31.5277 , numOfKernels*-14.1680,  numOfKernels* -5.7458,
                                       numOfKernels*-18.0441 , numOfKernels*-28.1205 , numOfKernels*-52.4022,  numOfKernels*-58.1271,  numOfKernels*-60.5645 , numOfKernels*-33.0438,  numOfKernels*-13.4845,
                                       numOfKernels*-23.4000 , numOfKernels*-44.9764 , numOfKernels*-63.5654,  numOfKernels*-67.3371,  numOfKernels*-67.1295 , numOfKernels*-42.8381,  numOfKernels*-19.9402,
                                       numOfKernels*-23.1051 , numOfKernels*-51.8992 , numOfKernels*-69.5876,  numOfKernels*-76.0211,  numOfKernels*-72.4573 , numOfKernels*-45.1977,  numOfKernels*-18.6720,
                                       numOfKernels*-13.0686 , numOfKernels*-27.3009 , numOfKernels*-32.8800,  numOfKernels*-35.8455,  numOfKernels*-33.8850 , numOfKernels*-19.7834,  numOfKernels* -7.2976,
                                       numOfKernels* -3.7941 , numOfKernels* -8.3188 , numOfKernels* -9.9258,  numOfKernels*-10.8878,  numOfKernels* -9.5107 , numOfKernels* -4.6338,  numOfKernels* -1.3574};

    std::vector<cmn::GPUFLOAT> tmp_reNewW = {12.4344,   14.2176,   13.2687,
                                 18.6388,   20.3466,   19.8105,
                                 21.0369,   24.4087,   23.4088};
    std::vector<cmn::GPUFLOAT> tmp_imNewW = {-15.6704,  -11.7766,  -14.5057,
                                 -11.3171,   -8.0477,  -10.3620,
                                  -6.6281,   -4.6506,   -4.8763};


    std::vector<cmn::GPUFLOAT> tmp_bias = {1,1};
    std::vector<cmn::GPUFLOAT> bias;
    std::vector<cmn::GPUFLOAT> weights;
    std::vector<cmn::GPUFLOAT> reResult;
    std::vector<cmn::GPUFLOAT> imResult;
    std::vector<cmn::GPUFLOAT> reInput;
    std::vector<cmn::GPUFLOAT> imInput;
    std::vector<cmn::GPUFLOAT> reNewW;
    std::vector<cmn::GPUFLOAT> imNewW;
    std::vector<cmn::GPUFLOAT> reReInput;
    std::vector<cmn::GPUFLOAT> imReInput;
    for(int i = 0; i < numOfKernels; ++i)
    {
        for(int i = 0; i < numOfChannels; ++i)
        {
            weights.insert(weights.end(), tmp_weights.begin(), tmp_weights.end());
            reNewW.insert(reNewW.end(), tmp_reNewW.begin(), tmp_reNewW.end());
            imNewW.insert(imNewW.end(), tmp_imNewW.begin(), tmp_imNewW.end());
        }
        bias.insert(bias.end(), tmp_bias.begin(), tmp_bias.end());
        reReInput.insert(reReInput.end(), tmp_reReInput.begin(), tmp_reReInput.end());
        imReInput.insert(imReInput.end(), tmp_imReInput.begin(), tmp_imReInput.end());
    }
    for(int i = 0; i < numOfChannels; ++i)
    {
        reResult.insert(reResult.end(), tmp_reResult.begin(), tmp_reResult.end());
        imResult.insert(imResult.end(), tmp_imResult.begin(), tmp_imResult.end());
        reInput.insert(reInput.end(), tmp_reInput.begin(), tmp_reInput.end());
        imInput.insert(imInput.end(), tmp_imInput.begin(), tmp_imInput.end());
    }
    const int inputSize  = 7;
    const int stride = 1;
    const int kernelSize = 3;
    conf::ConvLayerParams clayerParams(numOfChannels/*channels*/, numOfKernels/*kernels*/, kernelSize/*kernelsize*/, stride/*stride*/,
                                       2/*inputDim*/, inputSize/*inputSize*/, 0.1/*weightsdev*/, 0, conf::ConvLayerParams::WeightsType::COMPLEX, bias.front(), bias.back(), "fake"/*actFunc*/, "convtest"/*id*/);

    if(oclProgram.compile(oclContext))
    {
        //storage for result
        const unsigned int resSize = inputSize*inputSize;
        cmn::GPUFLOAT reCalculatedResult[numOfChannels*resSize];
        cmn::GPUFLOAT imCalculatedResult[numOfChannels*resSize];

        net::ConvLayer uut(clayerParams, weights, bias, envReader, oclProgram, oclContext);

        gpu::BufferIO input;
        //allocate enough memory on gpu (10 x input size should be enough)
        input.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, std::max(numOfChannels, numOfKernels)*sizeof(reCalculatedResult));
        input.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, std::max(numOfChannels, numOfKernels)*sizeof(imCalculatedResult));
        input.m_reShMemBkp = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, std::max(numOfChannels, numOfKernels)*sizeof(reCalculatedResult));
        input.m_imShMemBkp = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, std::max(numOfChannels, numOfKernels)*sizeof(imCalculatedResult));
        uut.writeToBuffer(oclContext.getCommandQueue(), input.m_reShMem, reInput.size()*sizeof(cmn::GPUFLOAT), reInput.data());
        uut.writeToBuffer(oclContext.getCommandQueue(), input.m_imShMem, imInput.size()*sizeof(cmn::GPUFLOAT), imInput.data());
        uut.setInput(input);
        uut.runFwdPropagation(oclContext);
        uut.printFwdProfilingInfo();

        //back propagation
        gpu::BufferIO bkpInput = uut.getOutput();
        uut.writeToBuffer(oclContext.getCommandQueue(), bkpInput.m_reShMem, reReInput.size()*sizeof(cmn::GPUFLOAT), reReInput.data());
        uut.writeToBuffer(oclContext.getCommandQueue(), bkpInput.m_imShMem, imReInput.size()*sizeof(cmn::GPUFLOAT), imReInput.data());

        uut.setBkpInput(bkpInput);
        uut.runBckPropagation(oclContext);
        uut.printBckProfilingInfo();

        gpu::BufferIO output = uut.getOutput();
        LayerWeights newWeights = uut.getWeights();
        cmn::GPUFLOAT reNewWeights[numOfChannels*kernelSize*kernelSize];
        cmn::GPUFLOAT imNewWeights[numOfChannels*kernelSize*kernelSize];
        //check result for each kernel
        for(int kernelId = 0; kernelId < numOfKernels; ++kernelId)
        {
            uut.readFromBuffer(oclContext.getCommandQueue(), newWeights[kernelId].m_re, sizeof(reNewWeights), reNewWeights);
            uut.readFromBuffer(oclContext.getCommandQueue(), newWeights[kernelId].m_im, sizeof(imNewWeights), imNewWeights);

            for(unsigned int channelId = 0; channelId < numOfChannels; ++channelId)
            {
                for(int k = 0; k < kernelSize*kernelSize; ++k)
                {
                    unsigned int kdx = kernelSize*kernelSize*channelId + k;
                    EXPECT_NEAR(reNewW[kernelId*kernelSize*kernelSize*numOfChannels + kdx], reNewWeights[kdx], 0.01);
                    EXPECT_NEAR(imNewW[kernelId*kernelSize*kernelSize*numOfChannels + kdx], imNewWeights[kdx], 0.01);
                }
            }
        }
        //check propagated error
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_reShMem, sizeof(reCalculatedResult), reCalculatedResult);
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_imShMem, sizeof(imCalculatedResult), imCalculatedResult);
        for(unsigned int channelId = 0; channelId < numOfChannels; ++channelId)
        {
            unsigned int idx = resSize*channelId;
            for(size_t i = 0; i < resSize; ++i)
            {
                EXPECT_NEAR(reResult[idx + i], reCalculatedResult[idx + i], 0.8);
                EXPECT_NEAR(imResult[idx + i], imCalculatedResult[idx + i], 0.8);
            }
        }
    }
}
