#include <gtest/gtest.h>
#include <ConvLayer.h>
#include <OpenCLContext.h>
#include <OpenCLProgram.h>
#include <OpenCLExecutionPlan.h>


using namespace neneta;
using namespace neneta::net;


extern conf::ConfigurationReader envReader;

void printComplexMatrix(cmn::GPUFLOAT reCalculatedResult[], cmn::GPUFLOAT imCalculatedResult[], int width)
{
    for(int i = 0; i < width; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            std::cout << reCalculatedResult[i*width+j] << " " << imCalculatedResult[i*width+j] << "j" << "\t";
        }
        std::cout << std::endl;
    }

}

TEST(ConvLayerTests, test_convolution)
{
    gpu::OpenCLContext oclContext(envReader);
    oclContext.printInfo();
    gpu::OpenCLProgram oclProgram(envReader);

    std::vector<cmn::GPUFLOAT> weights = {1,2,3, //repart
                                  4,5,6,
                                  7,8,9,
                                  1,2,3, //impart
                                  4,5,6,
                                  7,8,9};
    std::vector<cmn::GPUFLOAT> bias = {1,1};
    std::vector<cmn::GPUFLOAT> reInput = {0.2858, 0.5308, 0.3371, 0.2630, 0.9133, 0.1067, 0.3998,
                                0.7572, 0.7792, 0.1622, 0.6541, 0.1524, 0.9619, 0.2599,
                                0.7537, 0.9340, 0.7943, 0.6892, 0.8258, 0.0046, 0.8001,
                                0.3804, 0.1299, 0.3112, 0.7482, 0.5383, 0.7749, 0.4314,
                                0.5678, 0.5688, 0.5285, 0.4505, 0.9961, 0.8173, 0.9106,
                                0.0759, 0.4694, 0.1656, 0.0838, 0.0782, 0.8687, 0.1818,
                                0.0540, 0.0119, 0.6020, 0.2290, 0.4427, 0.0844, 0.2638};
    std::vector<cmn::GPUFLOAT> imInput = {0.1455, 0.6221, 0.1839, 0.4893, 0.2417, 0.0598, 0.6491,
                                0.1361, 0.3510, 0.2400, 0.3377, 0.4039, 0.2348, 0.7317,
                                0.8693, 0.5132, 0.4173, 0.9001, 0.0965, 0.3532, 0.6477,
                                0.5797, 0.4018, 0.0497, 0.3692, 0.1320, 0.8212, 0.4509,
                                0.5499, 0.0760, 0.9027, 0.1112, 0.9421, 0.0154, 0.5470,
                                0.1450, 0.2399, 0.9448, 0.7803, 0.9561, 0.0430, 0.2963,
                                0.8530, 0.1233, 0.4909, 0.3897, 0.5752, 0.1690, 0.7447};
  /* With "fake"  */
    std::vector<cmn::GPUFLOAT> reResult = {8.0338 , 4.0718 ,  5.8180 , 7.6914 , 6.5568 ,
                                 10.5100,  9.0989,   6.1444,  8.8156,  4.6000,
                                  4.1217,  7.5887,  11.5116,  7.5459,  9.1008,
                                 -1.5898,  3.1750,   3.2896,  8.8316,  7.8420,
                                 -3.4075, -1.7597, -12.0639,  3.0024,  6.8025};
    std::vector<cmn::GPUFLOAT> imResult = {38.3463, 40.9222, 36.1247, 37.6644, 37.2867,
                                 46.0207, 44.8634, 38.3401, 43.6203, 40.7205,
                                 50.6030, 48.6655, 47.8629, 47.0208, 45.7669,
                                 34.2677, 33.9790, 42.1219, 48.5566, 52.0058,
                                 38.6336, 37.6597, 51.5966, 44.5383, 51.5550};
    /* with complextanh */
 /*   std::vector<cmn::GPUFLOAT> reResult = { 1.0000,    1.0000,    1.0000,    1.0000,    1.0000,
                                    1.0000,    1.0000,    1.0000,    1.0000,    1.0000,
                                    1.0001,    1.0000,    1.0000,    1.0000,    1.0000,
                                   -0.7631,    0.9997,    1.0001,    1.0000,    1.0000,
                                   -1.0121,   -1.0607,   -1.0000,    1.0007,    1.0000};
    std::vector<cmn::GPUFLOAT> imResult = {-0.0000,    0.0001,   -0.0000,    0.0000,    0.0000,
                                   -0.0000,   -0.0000,   -0.0000,    0.0000,    0.0000,
                                    0.0000,   -0.0000,   -0.0000,    0.0000,   -0.0000,
                                    0.5122,    0.0004,   -0.0004,   -0.0000,   -0.0000,
                                   -0.0109,    0.4579,   -0.0000,    0.0000,   -0.0000};*/

    const int inputSize  = 7;
    const int stride = 1;
    const int kernelSize = 3;
    const int outputSize = (inputSize-kernelSize)/stride + 1;
    conf::ConvLayerParams clayerParams(1/*channels*/, 1/*kernels*/, kernelSize/*kernelsize*/, stride/*stride*/,
                                       2/*inputDim*/, inputSize/*inputSize*/, 0.1/*weightsdev*/, 0, conf::ConvLayerParams::WeightsType::COMPLEX, bias.front(), bias.back(), "fake"/*complextanh*/, "convtest"/*id*/);

    if(oclProgram.compile(oclContext))
    {

        net::ConvLayer uut(clayerParams, weights, bias, envReader, oclProgram, oclContext);

        gpu::BufferIO input;
        input.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, reInput.size()*sizeof(cmn::GPUFLOAT), reInput.data());
        input.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, imInput.size()*sizeof(cmn::GPUFLOAT), imInput.data());

        uut.setInput(input);

        uut.runFwdPropagation(oclContext);
        uut.printFwdProfilingInfo();

        gpu::BufferIO output = uut.getOutput();

        cmn::GPUFLOAT reCalculatedResult[outputSize*outputSize];
        cmn::GPUFLOAT imCalculatedResult[outputSize*outputSize];
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_reShMem, sizeof(reCalculatedResult), reCalculatedResult);
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_imShMem, sizeof(imCalculatedResult), imCalculatedResult);

        for(int i = 0; i < outputSize*outputSize; ++i)
        {
            EXPECT_NEAR(reResult[i], reCalculatedResult[i], 0.01);
            EXPECT_NEAR(imResult[i], imCalculatedResult[i], 0.01);
        }
        printComplexMatrix(reCalculatedResult, imCalculatedResult, outputSize);

    }
}

TEST(ConvLayerTests, test_convolution_stride)
{
    gpu::OpenCLContext oclContext(envReader);
    oclContext.printInfo();
    gpu::OpenCLProgram oclProgram(envReader);

    std::vector<cmn::GPUFLOAT> weights = {1,2,3, //repart
                                  4,5,6,
                                  7,8,9,
                                  1,2,3, //impart
                                  4,5,6,
                                  7,8,9};
    std::vector<cmn::GPUFLOAT> bias = {1,1};
    std::vector<cmn::GPUFLOAT> reInput = {0.2858, 0.5308, 0.3371, 0.2630, 0.9133, 0.1067, 0.3998,
                                0.7572, 0.7792, 0.1622, 0.6541, 0.1524, 0.9619, 0.2599,
                                0.7537, 0.9340, 0.7943, 0.6892, 0.8258, 0.0046, 0.8001,
                                0.3804, 0.1299, 0.3112, 0.7482, 0.5383, 0.7749, 0.4314,
                                0.5678, 0.5688, 0.5285, 0.4505, 0.9961, 0.8173, 0.9106,
                                0.0759, 0.4694, 0.1656, 0.0838, 0.0782, 0.8687, 0.1818,
                                0.0540, 0.0119, 0.6020, 0.2290, 0.4427, 0.0844, 0.2638};
    std::vector<cmn::GPUFLOAT> imInput = {0.1455, 0.6221, 0.1839, 0.4893, 0.2417, 0.0598, 0.6491,
                                0.1361, 0.3510, 0.2400, 0.3377, 0.4039, 0.2348, 0.7317,
                                0.8693, 0.5132, 0.4173, 0.9001, 0.0965, 0.3532, 0.6477,
                                0.5797, 0.4018, 0.0497, 0.3692, 0.1320, 0.8212, 0.4509,
                                0.5499, 0.0760, 0.9027, 0.1112, 0.9421, 0.0154, 0.5470,
                                0.1450, 0.2399, 0.9448, 0.7803, 0.9561, 0.0430, 0.2963,
                                0.8530, 0.1233, 0.4909, 0.3897, 0.5752, 0.1690, 0.7447};
    /*with fake act
    std::vector<cmn::GPUFLOAT> reResult = {8.0338  ,  5.8180 , 6.5568 ,
                                  4.1217,   11.5116,  9.1008,
                                 -3.4075,  -12.0639,  6.8025};
    std::vector<cmn::GPUFLOAT> imResult = {38.3463,  36.1247, 37.2867,
                                   50.6030,  47.8629, 45.7669,
                                   38.6336,  51.5966, 51.5550};*/
    /*with complextanh act*/
    std::vector<cmn::GPUFLOAT> reResult = { 1.0000,   1.0000,   1.0000,
                                    1.0001,   1.0000,   1.0000,
                                   -1.0121,  -1.0000,   1.0000};
    std::vector<cmn::GPUFLOAT> imResult = {-0.0000,  -0.0000,   0.0000,
                                    0.0000,  -0.0000,  -0.0000,
                                   -0.0109,  -0.0000,  -0.0000};

    const int inputSize  = 7;
    const int stride = 2;
    const int kernelSize = 3;
    const int outputSize = (inputSize-kernelSize)/stride + 1;
    conf::ConvLayerParams clayerParams(1/*channels*/, 1/*kernels*/, kernelSize/*kernelsize*/, stride/*stride*/,
                                       2/*inputDim*/, inputSize/*inputSize*/, 0.1/*weightsdev*/, 0,conf::ConvLayerParams::WeightsType::COMPLEX, bias.front(), bias.back(), "complextanh"/*actFunc*/, "convtest"/*id*/);

    if(oclProgram.compile(oclContext))
    {

        net::ConvLayer uut(clayerParams, weights, bias, envReader, oclProgram, oclContext);

        gpu::BufferIO input;
        input.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, reInput.size()*sizeof(cmn::GPUFLOAT), reInput.data());
        input.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, imInput.size()*sizeof(cmn::GPUFLOAT), imInput.data());

        uut.setInput(input);

        uut.runFwdPropagation(oclContext);
        uut.printFwdProfilingInfo();

        gpu::BufferIO output = uut.getOutput();

        cmn::GPUFLOAT reCalculatedResult[outputSize*outputSize];
        cmn::GPUFLOAT imCalculatedResult[outputSize*outputSize];
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_reShMem, sizeof(reCalculatedResult), reCalculatedResult);
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_imShMem, sizeof(imCalculatedResult), imCalculatedResult);

        for(int i = 0; i < outputSize*outputSize; ++i)
        {
            EXPECT_NEAR(reResult[i], reCalculatedResult[i], 0.01);
            EXPECT_NEAR(imResult[i], imCalculatedResult[i], 0.01);
        }
        printComplexMatrix(reCalculatedResult, imCalculatedResult, outputSize);

    }
}


TEST(ConvLayerTests, test_convolution_multichannel)
{
    const int numOfChannels = 200;
    gpu::OpenCLContext oclContext(envReader);
    oclContext.printInfo();
    gpu::OpenCLProgram oclProgram(envReader);

    std::vector<cmn::GPUFLOAT> tmpWeights = {1,2,3, //repart
                                  4,5,6,
                                  7,8,9,
                                  1,2,3, //impart
                                  4,5,6,
                                  7,8,9};
    std::vector<cmn::GPUFLOAT> bias = {1,1};
    std::vector<cmn::GPUFLOAT> tmpReInput = {0.2858, 0.5308, 0.3371, 0.2630, 0.9133, 0.1067, 0.3998,
                                0.7572, 0.7792, 0.1622, 0.6541, 0.1524, 0.9619, 0.2599,
                                0.7537, 0.9340, 0.7943, 0.6892, 0.8258, 0.0046, 0.8001,
                                0.3804, 0.1299, 0.3112, 0.7482, 0.5383, 0.7749, 0.4314,
                                0.5678, 0.5688, 0.5285, 0.4505, 0.9961, 0.8173, 0.9106,
                                0.0759, 0.4694, 0.1656, 0.0838, 0.0782, 0.8687, 0.1818,
                                0.0540, 0.0119, 0.6020, 0.2290, 0.4427, 0.0844, 0.2638};
    std::vector<cmn::GPUFLOAT> tmpImInput = {0.1455, 0.6221, 0.1839, 0.4893, 0.2417, 0.0598, 0.6491,
                                0.1361, 0.3510, 0.2400, 0.3377, 0.4039, 0.2348, 0.7317,
                                0.8693, 0.5132, 0.4173, 0.9001, 0.0965, 0.3532, 0.6477,
                                0.5797, 0.4018, 0.0497, 0.3692, 0.1320, 0.8212, 0.4509,
                                0.5499, 0.0760, 0.9027, 0.1112, 0.9421, 0.0154, 0.5470,
                                0.1450, 0.2399, 0.9448, 0.7803, 0.9561, 0.0430, 0.2963,
                                0.8530, 0.1233, 0.4909, 0.3897, 0.5752, 0.1690, 0.7447};
  /* With "fake" */
    std::vector<cmn::GPUFLOAT> reResult = {8.0338 , 4.0718 ,  5.8180 , 7.6914 , 6.5568 ,
                                 10.5100,  9.0989,   6.1444,  8.8156,  4.6000,
                                  4.1217,  7.5887,  11.5116,  7.5459,  9.1008,
                                 -1.5898,  3.1750,   3.2896,  8.8316,  7.8420,
                                 -3.4075, -1.7597, -12.0639,  3.0024,  6.8025};
    std::vector<cmn::GPUFLOAT> imResult = {38.3463, 40.9222, 36.1247, 37.6644, 37.2867,
                                 46.0207, 44.8634, 38.3401, 43.6203, 40.7205,
                                 50.6030, 48.6655, 47.8629, 47.0208, 45.7669,
                                 34.2677, 33.9790, 42.1219, 48.5566, 52.0058,
                                 38.6336, 37.6597, 51.5966, 44.5383, 51.5550};

    for(size_t i = 0; i < reResult.size(); ++i)
    {
        reResult[i] = reResult[i]*numOfChannels;
        imResult[i] = imResult[i]*numOfChannels;
    }

    std::vector<cmn::GPUFLOAT> weights;
    std::vector<cmn::GPUFLOAT> reInput;
    std::vector<cmn::GPUFLOAT> imInput;
    for(int i = 0; i < numOfChannels; ++i)
    {
        weights.insert(weights.end(), tmpWeights.begin(), tmpWeights.end());
        reInput.insert(reInput.end(), tmpReInput.begin(), tmpReInput.end());
        imInput.insert(imInput.end(), tmpImInput.begin(), tmpImInput.end());
    }

    const int inputSize  = 7;
    const int stride = 1;
    const int kernelSize = 3;
    const int outputSize = (inputSize-kernelSize)/stride + 1;
    conf::ConvLayerParams clayerParams(numOfChannels/*channels*/, 1/*kernels*/, kernelSize/*kernelsize*/, stride/*stride*/,
                                       2/*inputDim*/, inputSize/*inputSize*/, 0.1/*weightsdev*/, 0, conf::ConvLayerParams::WeightsType::COMPLEX, bias.front(), bias.back(), "fake"/*actFunc*/, "convtest"/*id*/);

    if(oclProgram.compile(oclContext))
    {

        net::ConvLayer uut(clayerParams, weights, bias, envReader, oclProgram, oclContext);

        gpu::BufferIO input;
        input.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, reInput.size()*sizeof(cmn::GPUFLOAT), reInput.data());
        input.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, imInput.size()*sizeof(cmn::GPUFLOAT), imInput.data());

        uut.setInput(input);

        uut.runFwdPropagation(oclContext);
        uut.printFwdProfilingInfo();

        gpu::BufferIO output = uut.getOutput();

        cmn::GPUFLOAT reCalculatedResult[outputSize*outputSize];
        cmn::GPUFLOAT imCalculatedResult[outputSize*outputSize];
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_reShMem, sizeof(reCalculatedResult), reCalculatedResult);
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_imShMem, sizeof(imCalculatedResult), imCalculatedResult);

        for(int i = 0; i < outputSize*outputSize; ++i)
        {
            EXPECT_NEAR(reResult[i], reCalculatedResult[i], 4.6);
            EXPECT_NEAR(imResult[i], imCalculatedResult[i], 4.6);
        }



    }
}

TEST(ConvLayerTests, test_convolution_multikernel)
{
    const int numOfKernels = 2000;
    gpu::OpenCLContext oclContext(envReader);
    oclContext.printInfo();
    gpu::OpenCLProgram oclProgram(envReader);

    std::vector<cmn::GPUFLOAT> tmpWeights = {1,2,3, //repart
                                  4,5,6,
                                  7,8,9,
                                  1,2,3, //impart
                                  4,5,6,
                                  7,8,9};
    std::vector<cmn::GPUFLOAT> reInput = {0.2858, 0.5308, 0.3371, 0.2630, 0.9133, 0.1067, 0.3998,
                                0.7572, 0.7792, 0.1622, 0.6541, 0.1524, 0.9619, 0.2599,
                                0.7537, 0.9340, 0.7943, 0.6892, 0.8258, 0.0046, 0.8001,
                                0.3804, 0.1299, 0.3112, 0.7482, 0.5383, 0.7749, 0.4314,
                                0.5678, 0.5688, 0.5285, 0.4505, 0.9961, 0.8173, 0.9106,
                                0.0759, 0.4694, 0.1656, 0.0838, 0.0782, 0.8687, 0.1818,
                                0.0540, 0.0119, 0.6020, 0.2290, 0.4427, 0.0844, 0.2638};
    std::vector<cmn::GPUFLOAT> imInput = {0.1455, 0.6221, 0.1839, 0.4893, 0.2417, 0.0598, 0.6491,
                                0.1361, 0.3510, 0.2400, 0.3377, 0.4039, 0.2348, 0.7317,
                                0.8693, 0.5132, 0.4173, 0.9001, 0.0965, 0.3532, 0.6477,
                                0.5797, 0.4018, 0.0497, 0.3692, 0.1320, 0.8212, 0.4509,
                                0.5499, 0.0760, 0.9027, 0.1112, 0.9421, 0.0154, 0.5470,
                                0.1450, 0.2399, 0.9448, 0.7803, 0.9561, 0.0430, 0.2963,
                                0.8530, 0.1233, 0.4909, 0.3897, 0.5752, 0.1690, 0.7447};
  /* With "fake" */
    std::vector<cmn::GPUFLOAT> tmpReResult = {8.0338 , 4.0718 ,  5.8180 , 7.6914 , 6.5568 ,
                                 10.5100,  9.0989,   6.1444,  8.8156,  4.6000,
                                  4.1217,  7.5887,  11.5116,  7.5459,  9.1008,
                                 -1.5898,  3.1750,   3.2896,  8.8316,  7.8420,
                                 -3.4075, -1.7597, -12.0639,  3.0024,  6.8025};
    std::vector<cmn::GPUFLOAT> tmpImResult = {38.3463, 40.9222, 36.1247, 37.6644, 37.2867,
                                 46.0207, 44.8634, 38.3401, 43.6203, 40.7205,
                                 50.6030, 48.6655, 47.8629, 47.0208, 45.7669,
                                 34.2677, 33.9790, 42.1219, 48.5566, 52.0058,
                                 38.6336, 37.6597, 51.5966, 44.5383, 51.5550};
    std::vector<cmn::GPUFLOAT> tmpBias = {1,1};
    std::vector<cmn::GPUFLOAT> bias;
    std::vector<cmn::GPUFLOAT> weights;
    std::vector<cmn::GPUFLOAT> reResult;
    std::vector<cmn::GPUFLOAT> imResult;
    for(int i = 0; i < numOfKernels; ++i)
    {
        weights.insert(weights.end(), tmpWeights.begin(), tmpWeights.end());
        reResult.insert(reResult.end(), tmpReResult.begin(), tmpReResult.end());
        imResult.insert(imResult.end(), tmpImResult.begin(), tmpImResult.end());
        bias.insert(bias.end(), tmpBias.begin(), tmpBias.end());
    }

    const int inputSize  = 7;
    const int stride = 1;
    const int kernelSize = 3;
    const int outputSize = (inputSize-kernelSize)/stride + 1;
    conf::ConvLayerParams clayerParams(1/*channels*/, numOfKernels/*kernels*/, kernelSize/*kernelsize*/, stride/*stride*/,
                                       2/*inputDim*/, inputSize/*inputSize*/, 0.1/*weightsdev*/, 0,conf::ConvLayerParams::WeightsType::COMPLEX, bias.front(), bias.back(), "fake"/*actFunc*/, "convtest"/*id*/);

    if(oclProgram.compile(oclContext))
    {

        net::ConvLayer uut(clayerParams, weights, bias, envReader, oclProgram, oclContext);

        cmn::GPUFLOAT reCalculatedResult[numOfKernels*outputSize*outputSize];
        cmn::GPUFLOAT imCalculatedResult[numOfKernels*outputSize*outputSize];

        gpu::BufferIO input;
        input.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, sizeof(reCalculatedResult));
        input.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, sizeof(imCalculatedResult));
        uut.writeToBuffer(oclContext.getCommandQueue(), input.m_reShMem, reInput.size()*sizeof(cmn::GPUFLOAT), reInput.data());
        uut.writeToBuffer(oclContext.getCommandQueue(), input.m_imShMem, imInput.size()*sizeof(cmn::GPUFLOAT), imInput.data());

        uut.setInput(input);

        uut.runFwdPropagation(oclContext);
        uut.printFwdProfilingInfo();

        gpu::BufferIO output = uut.getOutput();


        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_reShMem, sizeof(reCalculatedResult), reCalculatedResult);
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_imShMem, sizeof(imCalculatedResult), imCalculatedResult);

        for(int kn = 0; kn < numOfKernels; ++kn)
        {
            for(int i = 0; i < outputSize*outputSize; ++i)
            {
                EXPECT_NEAR(reResult[kn*outputSize*outputSize+i], reCalculatedResult[kn*outputSize*outputSize+i], 0.1);
                EXPECT_NEAR(imResult[kn*outputSize*outputSize+i], imCalculatedResult[kn*outputSize*outputSize+i], 0.1);
            }
        }
    }
}
