#include <gtest/gtest.h>
#include <ConvLayer.h>
#include <OpenCLContext.h>
#include <OpenCLProgram.h>
#include <OpenCLExecutionPlan.h>


using namespace neneta;
using namespace neneta::net;


extern conf::ConfigurationReader envReader;
void printComplexMatrix(cmn::GPUFLOAT reCalculatedResult[], cmn::GPUFLOAT imCalculatedResult[], int width);

class ConvLayerBpTests : public ::testing::Test
{
protected:
    ConvLayerBpTests()
        : oclContext(envReader)
        , oclProgram(envReader)
        , inputSize(7)
        , stride(1)
        , kernelSize(3)
        , outputSize((inputSize-kernelSize)/stride + 1)
        , actFunc("fake")
        , clayerParams(1, 1, kernelSize, stride, 2, inputSize, 0.1, 0, conf::ConvLayerParams::WeightsType::COMPLEX, 1, 0, actFunc, "convtest")
        , weights({1,2,3, //repart
                    4,5,6,
                    7,8,9,
                    1,2,3, //impart
                    4,5,6,
                    7,8,9})
        , bias({1,1})
        , reInput({0.2858, 0.5308, 0.3371, 0.2630, 0.9133, 0.1067, 0.3998,
                  0.7572, 0.7792, 0.1622, 0.6541, 0.1524, 0.9619, 0.2599,
                  0.7537, 0.9340, 0.7943, 0.6892, 0.8258, 0.0046, 0.8001,
                  0.3804, 0.1299, 0.3112, 0.7482, 0.5383, 0.7749, 0.4314,
                  0.5678, 0.5688, 0.5285, 0.4505, 0.9961, 0.8173, 0.9106,
                  0.0759, 0.4694, 0.1656, 0.0838, 0.0782, 0.8687, 0.1818,
                  0.0540, 0.0119, 0.6020, 0.2290, 0.4427, 0.0844, 0.2638,
                  0.2858, 0.5308, 0.3371, 0.2630, 0.9133, 0.1067, 0.3998,
                0.7572, 0.7792, 0.1622, 0.6541, 0.1524, 0.9619, 0.2599,//rubish
                0.7537, 0.9340, 0.7943, 0.6892, 0.8258, 0.0046, 0.8001,
                0.3804, 0.1299, 0.3112, 0.7482, 0.5383, 0.7749, 0.4314,
                0.5678, 0.5688, 0.5285, 0.4505, 0.9961, 0.8173, 0.9106,
                0.0759, 0.4694, 0.1656, 0.0838, 0.0782, 0.8687, 0.1818,
                0.0540, 0.0119, 0.6020, 0.2290, 0.4427, 0.0844, 0.2638})
        , imInput({0.1455, 0.6221, 0.1839, 0.4893, 0.2417, 0.0598, 0.6491,
                  0.1361, 0.3510, 0.2400, 0.3377, 0.4039, 0.2348, 0.7317,
                  0.8693, 0.5132, 0.4173, 0.9001, 0.0965, 0.3532, 0.6477,
                  0.5797, 0.4018, 0.0497, 0.3692, 0.1320, 0.8212, 0.4509,
                  0.5499, 0.0760, 0.9027, 0.1112, 0.9421, 0.0154, 0.5470,
                  0.1450, 0.2399, 0.9448, 0.7803, 0.9561, 0.0430, 0.2963,
                  0.8530, 0.1233, 0.4909, 0.3897, 0.5752, 0.1690, 0.7447,
                  0.2858, 0.5308, 0.3371, 0.2630, 0.9133, 0.1067, 0.3998,
                    0.7572, 0.7792, 0.1622, 0.6541, 0.1524, 0.9619, 0.2599, //rubish
                    0.7537, 0.9340, 0.7943, 0.6892, 0.8258, 0.0046, 0.8001,
                    0.3804, 0.1299, 0.3112, 0.7482, 0.5383, 0.7749, 0.4314,
                    0.5678, 0.5688, 0.5285, 0.4505, 0.9961, 0.8173, 0.9106,
                    0.0759, 0.4694, 0.1656, 0.0838, 0.0782, 0.8687, 0.1818,
                    0.0540, 0.0119, 0.6020, 0.2290, 0.4427, 0.0844, 0.2638})
        , uut(clayerParams, weights, bias, envReader, oclProgram, oclContext)
    {}

    void SetUp()
    {
        oclProgram.compile(oclContext);
        input.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, reInput.size()*sizeof(cmn::GPUFLOAT), reInput.data());
        input.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, imInput.size()*sizeof(cmn::GPUFLOAT), imInput.data());
        input.m_reShMemBkp = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, reInput.size()*sizeof(cmn::GPUFLOAT), reInput.data());
        input.m_imShMemBkp = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, imInput.size()*sizeof(cmn::GPUFLOAT), imInput.data());

        uut.setInput(input);

        uut.runFwdPropagation(oclContext);
        uut.printFwdProfilingInfo();
    }
    gpu::OpenCLContext oclContext;
    gpu::OpenCLProgram oclProgram;
    const int inputSize;
    const int stride;
    const int kernelSize;
    const int outputSize;
    std::string actFunc;
    conf::ConvLayerParams clayerParams;
    std::vector<cmn::GPUFLOAT> weights;
    std::vector<cmn::GPUFLOAT> bias;
    std::vector<cmn::GPUFLOAT> reInput;
    std::vector<cmn::GPUFLOAT> imInput;
    net::ConvLayer uut;
    gpu::BufferIO input;
};


TEST_F(ConvLayerBpTests, test_delta_updates)
{
  /* With "fake" all points in actder are 1+j1 */
    std::vector<cmn::GPUFLOAT> reReInput = { 0.8147, 0.0975, 0.1576, 0.1419, 0.6557,
                                   0.9058, 0.2785, 0.9706, 0.4218, 0.0357,
                                   0.1270, 0.5469, 0.9572, 0.9157, 0.8491,
                                   0.9134, 0.9575, 0.4854, 0.7922, 0.9340,
                                   0.6324, 0.9649, 0.8003, 0.9595, 0.6787};
    std::vector<cmn::GPUFLOAT> imReInput = { 0.7577, 0.7060, 0.8235, 0.4387, 0.4898,
                                   0.7431, 0.0318, 0.6948, 0.3816, 0.4456,
                                   0.3922, 0.2769, 0.3171, 0.7655, 0.6463,
                                   0.6555, 0.0462, 0.9502, 0.7952, 0.7094,
                                   0.1712, 0.0971, 0.0344, 0.1869, 0.7547};

    std::vector<cmn::GPUFLOAT> reResult = {1.5725, 0.8036, 0.9811, 0.5806, 1.1455,
                                   1.6489, 0.3103, 1.6654, 0.8033, 0.4813,
                                   0.5192, 0.8238, 1.2743, 1.6813, 1.4954,
                                   1.5689, 1.0037, 1.4356, 1.5874, 1.6434,
                                   0.8035, 1.0620, 0.8347, 1.1464, 1.4334};
    std::vector<cmn::GPUFLOAT> imResult = {-0.0570,  0.6085,  0.6658,  0.2969, -0.1660,
                                   -0.1627, -0.2467, -0.2758, -0.0402,  0.4099,
                                    0.2652, -0.2700, -0.6401, -0.1502, -0.2028,
                                   -0.2579, -0.9113,  0.4648,  0.0030, -0.2246,
                                   -0.4612, -0.8678, -0.7658, -0.7726,  0.0760};

    gpu::BufferIO bkpInput = uut.getOutput();
    uut.writeToBuffer(oclContext.getCommandQueue(), bkpInput.m_reShMem, reReInput.size()*sizeof(cmn::GPUFLOAT), reReInput.data());
    uut.writeToBuffer(oclContext.getCommandQueue(), bkpInput.m_imShMem, imReInput.size()*sizeof(cmn::GPUFLOAT), imReInput.data());

    uut.prepareBuffers(bkpInput);
    uut.calculateDeltas();
    uut.runBckPropagation(oclContext);
    uut.printBckProfilingInfo();

    LayerDeltas deltas = uut.getDeltas();

    cmn::GPUFLOAT reCalculatedResult[outputSize*outputSize];
    cmn::GPUFLOAT imCalculatedResult[outputSize*outputSize];
    uut.readFromBuffer(oclContext.getCommandQueue(), deltas.m_re, sizeof(reCalculatedResult), reCalculatedResult);
    uut.readFromBuffer(oclContext.getCommandQueue(), deltas.m_im, sizeof(imCalculatedResult), imCalculatedResult);

    for(int i = 0; i < outputSize*outputSize; ++i)
    {
        EXPECT_NEAR(reResult[i], reCalculatedResult[i], 0.01);
        EXPECT_NEAR(imResult[i], imCalculatedResult[i], 0.01);
    }
    //printComplexMatrix(reCalculatedResult, imCalculatedResult, outputSize);
}

TEST_F(ConvLayerBpTests, test_flip)
{
    std::vector<cmn::GPUFLOAT> reResult = { 0.2638, 0.0844, 0.4427, 0.2290, 0.6020, 0.0119, 0.0540,
                                    0.1818, 0.8687, 0.0782, 0.0838, 0.1656, 0.4694, 0.0759,
                                    0.9106, 0.8173, 0.9961, 0.4505, 0.5285, 0.5688, 0.5678,
                                    0.4314, 0.7749, 0.5383, 0.7482, 0.3112, 0.1299, 0.3804,
                                    0.8001, 0.0046, 0.8258, 0.6892, 0.7943, 0.9340, 0.7537,
                                    0.2599, 0.9619, 0.1524, 0.6541, 0.1622, 0.7792, 0.7572,
                                    0.3998, 0.1067, 0.9133, 0.2630, 0.3371, 0.5308, 0.2858};
    std::vector<cmn::GPUFLOAT> imResult =  {0.7447, 0.1690, 0.5752, 0.3897, 0.4909, 0.1233, 0.8530,
                                    0.2963, 0.0430, 0.9561, 0.7803, 0.9448, 0.2399, 0.1450,
                                    0.5470, 0.0154, 0.9421, 0.1112, 0.9027, 0.0760, 0.5499,
                                    0.4509, 0.8212, 0.1320, 0.3692, 0.0497, 0.4018, 0.5797,
                                    0.6477, 0.3532, 0.0965, 0.9001, 0.4173, 0.5132, 0.8693,
                                    0.7317, 0.2348, 0.4039, 0.3377, 0.2400, 0.3510, 0.1361,
                                    0.6491, 0.0598, 0.2417, 0.4893, 0.1839, 0.6221, 0.1455};

    uut.rotateInput();
    uut.runBckPropagation(oclContext);
    uut.printBckProfilingInfo();

    LayerInput input = uut.getLayerInput();

    cmn::GPUFLOAT reCalculatedResult[inputSize*inputSize];
    cmn::GPUFLOAT imCalculatedResult[inputSize*inputSize];
    uut.readFromBuffer(oclContext.getCommandQueue(), input.m_re, sizeof(reCalculatedResult), reCalculatedResult);
    uut.readFromBuffer(oclContext.getCommandQueue(), input.m_im, sizeof(imCalculatedResult), imCalculatedResult);

    for(int i = 0; i < inputSize*inputSize; ++i)
    {
        EXPECT_NEAR(reResult[i], reCalculatedResult[i], 0.01);
        EXPECT_NEAR(imResult[i], imCalculatedResult[i], 0.01);
    }
    //printComplexMatrix(reCalculatedResult, imCalculatedResult, inputSize);
}

TEST_F(ConvLayerBpTests, test_error_calculation)
{
    /* With "fake" all points in actder are 1+j1 */
      std::vector<cmn::GPUFLOAT> reReInput = { 0.8147, 0.0975, 0.1576, 0.1419, 0.6557,
                                     0.9058, 0.2785, 0.9706, 0.4218, 0.0357,
                                     0.1270, 0.5469, 0.9572, 0.9157, 0.8491,
                                     0.9134, 0.9575, 0.4854, 0.7922, 0.9340,
                                     0.6324, 0.9649, 0.8003, 0.9595, 0.6787};
      std::vector<cmn::GPUFLOAT> imReInput = { 0.7577, 0.7060, 0.8235, 0.4387, 0.4898,
                                     0.7431, 0.0318, 0.6948, 0.3816, 0.4456,
                                     0.3922, 0.2769, 0.3171, 0.7655, 0.6463,
                                     0.6555, 0.0462, 0.9502, 0.7952, 0.7094,
                                     0.1712, 0.0971, 0.0344, 0.1869, 0.7547};

      std::vector<cmn::GPUFLOAT> reResultE =      {13.6395 ,  24.8329 ,  36.7274 ,  30.9574,   27.3638,   13.9785,    6.8565,
                                           22.4688 ,  28.5121 ,  46.4225 ,  37.5778,   40.7049,   20.8788,   10.1564,
                                           20.5233 ,  26.3393 ,  39.5098 ,  41.8495,   49.3831,   31.4034,   13.5925,
                                           20.9640 ,  21.7276 ,  42.5138 ,  49.8685,   62.3325,   37.6161,   15.9932,
                                           13.2999 ,  14.8258 ,  25.4728 ,  31.1043,   48.6979,   32.2637,   17.5336,
                                            5.9868 ,   5.7759 ,   9.9506 ,  12.0285,   20.5386,   13.4702,    7.4564,
                                            1.0269 ,   1.2672 ,   0.9374 ,   1.4534,    5.3447,    3.3926,    1.5094};
      std::vector<cmn::GPUFLOAT> imResultE =     {-14.6655 , -14.7919 , -15.8050,   -6.4414,  -16.2802 , -12.4779,   -9.1805,
                                          -26.0814 , -28.8239 , -43.9933,  -31.0792,  -31.5277 , -14.1680,   -5.7458,
                                          -18.0441 , -28.1205 , -52.4022,  -58.1271,  -60.5645 , -33.0438,  -13.4845,
                                          -23.4000 , -44.9764 , -63.5654,  -67.3371,  -67.1295 , -42.8381,  -19.9402,
                                          -23.1051 , -51.8992 , -69.5876,  -76.0211,  -72.4573 , -45.1977,  -18.6720,
                                          -13.0686 , -27.3009 , -32.8800,  -35.8455,  -33.8850 , -19.7834,   -7.2976,
                                           -3.7941 ,  -8.3188 ,  -9.9258,  -10.8878,   -9.5107 ,  -4.6338,   -1.3574};



    gpu::BufferIO bkpInput = uut.getOutput();
    uut.writeToBuffer(oclContext.getCommandQueue(), bkpInput.m_reShMem, reReInput.size()*sizeof(cmn::GPUFLOAT), reReInput.data());
    uut.writeToBuffer(oclContext.getCommandQueue(), bkpInput.m_imShMem, imReInput.size()*sizeof(cmn::GPUFLOAT), imReInput.data());

    uut.prepareBuffers(bkpInput);
    uut.calculateDeltas();
    uut.rotateInput();    
    uut.calculateErrors(0);
    uut.runBckPropagation(oclContext);
    uut.printBckProfilingInfo();

    gpu::BufferIO output = uut.getOutput();

    cmn::GPUFLOAT reCalculatedResult[inputSize*inputSize];
    cmn::GPUFLOAT imCalculatedResult[inputSize*inputSize];
    uut.readFromBuffer(oclContext.getCommandQueue(), output.m_reShMem, sizeof(reCalculatedResult), reCalculatedResult);
    uut.readFromBuffer(oclContext.getCommandQueue(), output.m_imShMem, sizeof(imCalculatedResult), imCalculatedResult);

    for(int i = 0; i < inputSize*inputSize; ++i)
    {
        EXPECT_NEAR(reResultE[i], reCalculatedResult[i], 0.01);
        EXPECT_NEAR(imResultE[i], imCalculatedResult[i], 0.01);
    }
    //printComplexMatrix(reCalculatedResult, imCalculatedResult, inputSize);
}

TEST_F(ConvLayerBpTests, test_weights_update)
{
    /* With "fake" all points in actder are 1+j1 */
      std::vector<cmn::GPUFLOAT> reReInput = { 0.8147, 0.0975, 0.1576, 0.1419, 0.6557,
                                     0.9058, 0.2785, 0.9706, 0.4218, 0.0357,
                                     0.1270, 0.5469, 0.9572, 0.9157, 0.8491,
                                     0.9134, 0.9575, 0.4854, 0.7922, 0.9340,
                                     0.6324, 0.9649, 0.8003, 0.9595, 0.6787};
      std::vector<cmn::GPUFLOAT> imReInput = { 0.7577, 0.7060, 0.8235, 0.4387, 0.4898,
                                     0.7431, 0.0318, 0.6948, 0.3816, 0.4456,
                                     0.3922, 0.2769, 0.3171, 0.7655, 0.6463,
                                     0.6555, 0.0462, 0.9502, 0.7952, 0.7094,
                                     0.1712, 0.0971, 0.0344, 0.1869, 0.7547};


      std::vector<cmn::GPUFLOAT> reResultE =      {13.6395 ,  24.8329 ,  36.7274 ,  30.9574,   27.3638,   13.9785,    6.8565,
                                           22.4688 ,  28.5121 ,  46.4225 ,  37.5778,   40.7049,   20.8788,   10.1564,
                                           20.5233 ,  26.3393 ,  39.5098 ,  41.8495,   49.3831,   31.4034,   13.5925,
                                           20.9640 ,  21.7276 ,  42.5138 ,  49.8685,   62.3325,   37.6161,   15.9932,
                                           13.2999 ,  14.8258 ,  25.4728 ,  31.1043,   48.6979,   32.2637,   17.5336,
                                            5.9868 ,   5.7759 ,   9.9506 ,  12.0285,   20.5386,   13.4702,    7.4564,
                                            1.0269 ,   1.2672 ,   0.9374 ,   1.4534,    5.3447,    3.3926,    1.5094};
      std::vector<cmn::GPUFLOAT> imResultE =     {-14.6655 , -14.7919 , -15.8050,   -6.4414,  -16.2802 , -12.4779,   -9.1805,
                                          -26.0814 , -28.8239 , -43.9933,  -31.0792,  -31.5277 , -14.1680,   -5.7458,
                                          -18.0441 , -28.1205 , -52.4022,  -58.1271,  -60.5645 , -33.0438,  -13.4845,
                                          -23.4000 , -44.9764 , -63.5654,  -67.3371,  -67.1295 , -42.8381,  -19.9402,
                                          -23.1051 , -51.8992 , -69.5876,  -76.0211,  -72.4573 , -45.1977,  -18.6720,
                                          -13.0686 , -27.3009 , -32.8800,  -35.8455,  -33.8850 , -19.7834,   -7.2976,
                                           -3.7941 ,  -8.3188 ,  -9.9258,  -10.8878,   -9.5107 ,  -4.6338,   -1.3574};

      std::vector<cmn::GPUFLOAT> reNewW =  {   12.4344,   14.2176,   13.2687,
                                       18.6388,   20.3466,   19.8105,
                                       21.0369,   24.4087,   23.4088};
      std::vector<cmn::GPUFLOAT> imNewW =   {  -15.6704,  -11.7766,  -14.5057,
                                       -11.3171,   -8.0477,  -10.3620,
                                        -6.6281,   -4.6506,   -4.8763};



    gpu::BufferIO bkpInput = uut.getOutput();
    uut.writeToBuffer(oclContext.getCommandQueue(), bkpInput.m_reShMem, reReInput.size()*sizeof(cmn::GPUFLOAT), reReInput.data());
    uut.writeToBuffer(oclContext.getCommandQueue(), bkpInput.m_imShMem, imReInput.size()*sizeof(cmn::GPUFLOAT), imReInput.data());

    uut.setBkpInput(bkpInput);
    uut.runBckPropagation(oclContext);
    uut.printBckProfilingInfo();

    gpu::BufferIO output = uut.getOutput();

    cmn::GPUFLOAT reCalculatedResult[inputSize*inputSize];
    cmn::GPUFLOAT imCalculatedResult[inputSize*inputSize];
    uut.readFromBuffer(oclContext.getCommandQueue(), output.m_reShMem, sizeof(reCalculatedResult), reCalculatedResult);
    uut.readFromBuffer(oclContext.getCommandQueue(), output.m_imShMem, sizeof(imCalculatedResult), imCalculatedResult);

    for(int i = 0; i < inputSize*inputSize; ++i)
    {
        EXPECT_NEAR(reResultE[i], reCalculatedResult[i], 0.01);
        EXPECT_NEAR(imResultE[i], imCalculatedResult[i], 0.01);
    }
    //printComplexMatrix(reCalculatedResult, imCalculatedResult, inputSize);

    LayerWeights newWeights = uut.getWeights();

    cmn::GPUFLOAT reNewWeights[kernelSize*kernelSize];
    cmn::GPUFLOAT imNewWeights[kernelSize*kernelSize];
    uut.readFromBuffer(oclContext.getCommandQueue(), newWeights.front().m_re, sizeof(reNewWeights), reNewWeights);
    uut.readFromBuffer(oclContext.getCommandQueue(), newWeights.front().m_im, sizeof(imNewWeights), imNewWeights);

    for(int i = 0; i < kernelSize*kernelSize; ++i)
    {
        EXPECT_NEAR(reNewW[i], reNewWeights[i], 0.01);
        EXPECT_NEAR(imNewW[i], imNewWeights[i], 0.01);
    }
    //printComplexMatrix(reNewWeights, imNewWeights, kernelSize);
}
