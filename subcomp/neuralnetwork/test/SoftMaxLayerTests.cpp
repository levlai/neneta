#include <gtest/gtest.h>
#include <SoftMaxLayer.h>
#include <OpenCLContext.h>
#include <OpenCLProgram.h>
#include <OpenCLExecutionPlan.h>
#include <ErrorCalculationLayer.h>
#include <Image.h>
#include <gnuplot-iostream/gnuplot-iostream.h>
#include <ConfigurationReader.h>
#include <Plotter.h>

using namespace neneta;
using namespace neneta::net;


extern conf::ConfigurationReader envReader;

TEST(SoftMaxLayerTests, gnuplot)
{
    plot::Plotter plotter(envReader);
    plotter.plotNewPoint(1);
    plotter.plotNewPoint(2);
}

TEST(SoftMaxLayerTests, visualize_layer_weights)
{
    pers::Persistance persistance(envReader);
    std::vector<cmn::GPUFLOAT> blob;
    persistance.restoreFloatBlob("sm1", blob);

    unsigned int outputSize = 10;
    unsigned int inputSize = 28*28;
    for(size_t offset = 0; offset < inputSize*outputSize; offset += inputSize)
    {
        ip::Image<cmn::GPUFLOAT, std::int16_t> img(blob.data()+offset, 28);
        img.getCImg().display();
    }
}

TEST(SoftMaxLayerTests, softmaxlayer_forward_propagation_test_1000_outputsize)
{
    gpu::OpenCLContext oclContext(envReader);
    gpu::OpenCLProgram oclProgram(envReader);

    const unsigned int inputsize = 1000;
    const unsigned int outputsize = 1000;

    cmn::GPUFLOAT re = 0.00001;
    cmn::GPUFLOAT bias = 1;
    std::vector<cmn::GPUFLOAT> redata(inputsize, 0);
    std::vector<cmn::GPUFLOAT> weights(2*inputsize*outputsize, 0);
    for(unsigned int i = 0; i < 2*inputsize*outputsize; ++i)
    {
        weights[i] = static_cast<cmn::GPUFLOAT>(i)/(2*inputsize*outputsize);
    }

    for(unsigned int i = 0; i < inputsize; ++i)
    {
        redata[i] = re*i;
    }

    std::vector<cmn::GPUFLOAT> imdata(inputsize, 0);
    std::vector<cmn::GPUFLOAT> expected(outputsize, 0);
    cmn::GPUFLOAT naz = 0;
    for(unsigned int i = 0; i < outputsize; ++i)
    {
        cmn::GPUFLOAT act = 0;
        for(unsigned int j = 0; j < inputsize; ++j)
        {
            act += weights[2*i*outputsize+j]*redata[j];
        }
        act += bias;
        naz += exp(act);
        expected[i] = exp(act);
    }

    for(unsigned int i = 0; i < outputsize; ++i)
    {
        expected[i] /= naz;
    }

    conf::SoftMaxParams params(1, 1, inputsize, outputsize, "softmax", 0.1, 0, 1, "SoftMaxTest");

    if(oclProgram.compile(oclContext))
    {

        net::SoftMaxLayer uut(params, weights, envReader, oclProgram, oclContext);

        gpu::BufferIO input;
        input.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, inputsize*sizeof(cmn::GPUFLOAT), redata.data());
        input.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, inputsize*sizeof(cmn::GPUFLOAT), imdata.data());
        input.m_reShMemBkp = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, (inputsize+2)*outputsize*sizeof(cmn::GPUFLOAT));
        input.m_imShMemBkp = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, (inputsize+2)*outputsize*sizeof(cmn::GPUFLOAT));

        uut.setInput(input);

        uut.runFwdPropagation(oclContext);
        uut.printFwdProfilingInfo();

        gpu::BufferIO output = uut.getOutput();        

        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_reShMem, sizeof(cmn::GPUFLOAT)*outputsize, redata.data());
        for(unsigned int i = 0; i < outputsize; ++i)
        {
            EXPECT_NEAR(expected[i],redata[i],0.001);
        }
    }
}


TEST(SoftMaxLayerTests, softmaxlayer_forward_and_back_propagation_test_1000_inputsize)
{
    gpu::OpenCLContext oclContext(envReader);
    gpu::OpenCLProgram oclProgram(envReader);

    const unsigned int inputsize = 1000;
    const unsigned int outputsize = 1000;

    cmn::GPUFLOAT re = 0.00001;
    cmn::GPUFLOAT bias = 1;
    std::vector<cmn::GPUFLOAT> redata(inputsize, 0);
    std::vector<cmn::GPUFLOAT> weights(2*inputsize*outputsize, 0.1);
    for(unsigned int i = 0; i < 2*inputsize*outputsize; ++i)
    {
       weights[i] = static_cast<cmn::GPUFLOAT>(i+1)/(2*inputsize*outputsize);
    }
    std::random_shuffle(weights.begin(), weights.end());
    for(unsigned int i = 0; i < inputsize; ++i)
    {
        redata[i] = re*i;
    }

    std::vector<cmn::GPUFLOAT> imdata(inputsize, 0);
    std::vector<cmn::GPUFLOAT> expected(outputsize, 0);
    cmn::GPUFLOAT naz = 0;
    for(unsigned int i = 0; i < outputsize; ++i)
    {
        cmn::GPUFLOAT act = 0;
        for(unsigned int j = 0; j < inputsize; ++j)
        {
            act += weights[j+2*i*outputsize]*redata[j];
        }
        act += bias;
        naz += exp(act);
        expected[i] = exp(act);
    }

    std::vector<cmn::GPUFLOAT> redes(outputsize, 0);
    redes[0] = 1;
    std::vector<cmn::GPUFLOAT> imdes(outputsize, 0);

    std::vector<cmn::GPUFLOAT> expectedErrRe(outputsize, 1);
    for(unsigned int i = 0; i < outputsize; ++i)
    {
        expected[i] /= naz;
        expectedErrRe[i] = expected[i] - redes[i];

    }

    std::vector<cmn::GPUFLOAT> expectedBkp(inputsize, 0);
    for(unsigned int j = 0; j < inputsize; ++j)
    {
        for(unsigned int i = 0; i < outputsize; ++i)
        {
              expectedBkp[j] += expectedErrRe[i]*weights[2*i*outputsize+j];
        }
    }

    conf::SoftMaxParams params(1, 1, inputsize, outputsize, "softmax", 0.1, 0, 1, "SoftMaxTest");
    if(oclProgram.compile(oclContext))
    {
        conf::ErrorCalculationLayerParams errCalcParams(outputsize, "crossentropy", "smlayertest");
        net::ErrorCalculationLayer err1(errCalcParams, envReader, oclProgram, oclContext);

        net::SoftMaxLayer uut(params, weights, envReader, oclProgram, oclContext);

        gpu::BufferIO input;
        input.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, inputsize*sizeof(cmn::GPUFLOAT), redata.data());
        input.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, inputsize*sizeof(cmn::GPUFLOAT), imdata.data());
        input.m_reShMemBkp = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, (inputsize+2)*outputsize*sizeof(cmn::GPUFLOAT));
        input.m_imShMemBkp = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, (inputsize+2)*outputsize*sizeof(cmn::GPUFLOAT));
        input.m_reDesired = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cmn::GPUFLOAT)*outputsize, redes.data());
        input.m_imDesired = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cmn::GPUFLOAT)*outputsize, imdes.data());

        uut.setInput(input);

        uut >> err1;

        err1.runFwdPropagation(oclContext);
        err1.printFwdProfilingInfo();

        gpu::BufferIO output = err1.getOutput();

        err1.readFromBuffer(oclContext.getCommandQueue(), output.m_reShMem, sizeof(cmn::GPUFLOAT)*outputsize, redata.data());
        for(unsigned int i = 0; i < outputsize; ++i)
        {
            EXPECT_NEAR(expectedErrRe[i],redata[i],0.0001);
        }

        uut << err1;
        uut.runBckPropagation(oclContext);
        uut.printBckProfilingInfo();

        output = uut.getBkpOutput();

        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_reShMem, sizeof(cmn::GPUFLOAT)*inputsize, redata.data());
        for(unsigned int i = 0; i < inputsize; ++i)
        {
            EXPECT_NEAR(expectedBkp[i],redata[i],0.001);
        }
    }
}

