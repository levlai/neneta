#include <gtest/gtest.h>
#include <FCLayer.h>
#include <OpenCLContext.h>
#include <OpenCLProgram.h>
#include <OpenCLExecutionPlan.h>
#include <ErrorCalculationLayer.h>

using namespace neneta;
using namespace neneta::net;


extern conf::ConfigurationReader envReader;

TEST(FCLayerTests, fclayer_forward_propagation_test_one_channel_wgs)
{
    gpu::OpenCLContext oclContext(envReader);
    gpu::OpenCLProgram oclProgram(envReader);

    const unsigned int l = oclContext.getDevice().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    const int outputSize = 1;

    cmn::GPUFLOAT redata[l];
    cmn::GPUFLOAT imdata[l];
    std::vector<cmn::GPUFLOAT> weights(2*l, 1.0);
    std::vector<cmn::GPUFLOAT> bias = {1, 1};
    for(unsigned int i = 0; i < l; ++i)
    {
        redata[i] = 1.0;
        imdata[i] = 0;
    }

    conf::FCParams params(1, 1, l , outputSize, 0.1, 0, conf::FCParams::WeightsType::COMPLEX, 1, 0, "fake", "FCLayerTest");

    if(oclProgram.compile(oclContext))
    {

        net::FCLayer uut(params, weights, bias, envReader, oclProgram, oclContext);

        gpu::BufferIO input;
        input.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, l*sizeof(cmn::GPUFLOAT), redata);
        input.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, l*sizeof(cmn::GPUFLOAT), imdata);
        cl_buffer_region bg{0, sizeof(cmn::GPUFLOAT)*l};
        input.m_reChannels.emplace_back(input.m_reShMem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &bg));
        input.m_imChannels.emplace_back(input.m_imShMem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &bg));

        uut.setInput(input);

        uut.runFwdPropagation(oclContext);
        uut.printFwdProfilingInfo();

        gpu::BufferIO output = uut.getOutput();

        cmn::GPUFLOAT sum = 0;
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_reChannels.front(), sizeof(cmn::GPUFLOAT), &sum);
        ASSERT_EQ(sum, l+1);
        sum = 0;
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_imChannels.front(), sizeof(cmn::GPUFLOAT), &sum);
        ASSERT_EQ(sum, l+1);
    }
}

TEST(FCLayerTests, fclayer_forward_propagation_test_one_channel_lt_wgs)
{
    gpu::OpenCLContext oclContext(envReader);
    gpu::OpenCLProgram oclProgram(envReader);

    const unsigned int l = oclContext.getDevice().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()-9;
    const int outputSize = 1;
    cmn::GPUFLOAT redata[l];
    cmn::GPUFLOAT imdata[l];
    std::vector<cmn::GPUFLOAT> weights(2*l, 1.0);
    std::vector<cmn::GPUFLOAT> bias = {1, 1};
    for(unsigned int i = 0; i < l; ++i)
    {
        redata[i] = 1.0;
        imdata[i] = 0;
    }

    conf::FCParams params(1, 1, l, outputSize, 0.1, 0, conf::FCParams::WeightsType::COMPLEX, 1, 0, "fake", "FCLayerTest");

    if(oclProgram.compile(oclContext))
    {

        net::FCLayer uut(params, weights, bias, envReader, oclProgram, oclContext);

        gpu::BufferIO input;
        input.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, l*sizeof(cmn::GPUFLOAT), redata);
        input.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, l*sizeof(cmn::GPUFLOAT), imdata);
        cl_buffer_region bg{0, sizeof(cmn::GPUFLOAT)*l};
        input.m_reChannels.emplace_back(input.m_reShMem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &bg));
        input.m_imChannels.emplace_back(input.m_imShMem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &bg));

        uut.setInput(input);

        uut.runFwdPropagation(oclContext);
        uut.printFwdProfilingInfo();

        gpu::BufferIO output = uut.getOutput();

        cmn::GPUFLOAT sum = 0;
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_reChannels.front(), sizeof(cmn::GPUFLOAT), &sum);
        ASSERT_EQ(sum, l+1);
        sum = 0;
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_imChannels.front(), sizeof(cmn::GPUFLOAT), &sum);
        ASSERT_EQ(sum, l+1);
    }
}

TEST(FCLayerTests, fclayer_forward_propagation_test_one_channel_bt_wgs)
{
    gpu::OpenCLContext oclContext(envReader);
    gpu::OpenCLProgram oclProgram(envReader);

    const unsigned int l = oclContext.getDevice().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()+1;
    const int outputSize = 1;
    cmn::GPUFLOAT redata[l];
    cmn::GPUFLOAT imdata[l];
    std::vector<cmn::GPUFLOAT> weights(2*l, 1.0);
    std::vector<cmn::GPUFLOAT> bias = {1, 1};
    for(unsigned int i = 0; i < l; ++i)
    {
        redata[i] = 1.0;
        imdata[i] = 0;
    }

    conf::FCParams params(1, 1, l, outputSize, 0.1, 0, conf::FCParams::WeightsType::COMPLEX, 1, 0, "fake", "FCLayerTest");

    if(oclProgram.compile(oclContext))
    {

        net::FCLayer uut(params, weights, bias, envReader, oclProgram, oclContext);

        gpu::BufferIO input;
        input.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, l*sizeof(cmn::GPUFLOAT), redata);
        input.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, l*sizeof(cmn::GPUFLOAT), imdata);
        cl_buffer_region bg{0, sizeof(cmn::GPUFLOAT)*l};
        input.m_reChannels.emplace_back(input.m_reShMem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &bg));
        input.m_imChannels.emplace_back(input.m_imShMem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &bg));

        uut.setInput(input);

        uut.runFwdPropagation(oclContext);
        uut.printFwdProfilingInfo();

        gpu::BufferIO output = uut.getOutput();

        cmn::GPUFLOAT sum = 0;
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_reChannels.front(), sizeof(cmn::GPUFLOAT), &sum);
        ASSERT_EQ(sum, l+1);
        sum = 0;
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_imChannels.front(), sizeof(cmn::GPUFLOAT), &sum);
        ASSERT_EQ(sum, l+1);
    }
}

TEST(FCLayerTests, fclayer_forward_propagation_test_one_channel_500_wgs)
{
    gpu::OpenCLContext oclContext(envReader);
    gpu::OpenCLProgram oclProgram(envReader);

    const unsigned int l = 500*oclContext.getDevice().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    const int outputSize = 1;
    cmn::GPUFLOAT redata[l];
    cmn::GPUFLOAT imdata[l];
    std::vector<cmn::GPUFLOAT> weights(2*l, 1.0);
    std::vector<cmn::GPUFLOAT> bias = {1, 1};
    for(unsigned int i = 0; i < l; ++i)
    {
        redata[i] = 1.0;
        imdata[i] = 0;
    }

    conf::FCParams params(1, 1, l, outputSize, 0.1, 0, conf::FCParams::WeightsType::COMPLEX, 1, 0, "fake", "FCLayerTest");

    if(oclProgram.compile(oclContext))
    {

        net::FCLayer uut(params, weights, bias, envReader, oclProgram, oclContext);

        gpu::BufferIO input;
        input.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, l*sizeof(cmn::GPUFLOAT), redata);
        input.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, l*sizeof(cmn::GPUFLOAT), imdata);
        cl_buffer_region bg{0, sizeof(cmn::GPUFLOAT)*l};
        input.m_reChannels.emplace_back(input.m_reShMem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &bg));
        input.m_imChannels.emplace_back(input.m_imShMem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &bg));

        uut.setInput(input);

        uut.runFwdPropagation(oclContext);
        uut.printFwdProfilingInfo();

        gpu::BufferIO output = uut.getOutput();

        cmn::GPUFLOAT sum = 0;
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_reChannels.front(), sizeof(cmn::GPUFLOAT), &sum);
        ASSERT_EQ(sum, l+1);
        sum = 0;
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_imChannels.front(), sizeof(cmn::GPUFLOAT), &sum);
        ASSERT_EQ(sum, l+1);
    }
}


TEST(FCLayerTests, fclayer_forward_propagation_test_one_channel_2d_wgs)
{
    gpu::OpenCLContext oclContext(envReader);
    gpu::OpenCLProgram oclProgram(envReader);

    const unsigned int l = std::pow(oclContext.getDevice().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(),2);
    const int outputSize = 1;
    cmn::GPUFLOAT redata[l];
    cmn::GPUFLOAT imdata[l];
    std::vector<cmn::GPUFLOAT> weights(2*l, 1.0);
    std::vector<cmn::GPUFLOAT> bias = {1, 1};
    for(unsigned int i = 0; i < l; ++i)
    {
        redata[i] = 1.0;
        imdata[i] = 0;
    }

    conf::FCParams params(1, 2, sqrt(l), outputSize, 0.1, 0, conf::FCParams::WeightsType::COMPLEX, 1, 0, "fake", "FCLayerTest");

    if(oclProgram.compile(oclContext))
    {

        net::FCLayer uut(params, weights, bias, envReader, oclProgram, oclContext);

        gpu::BufferIO input;
        input.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, l*sizeof(cmn::GPUFLOAT), redata);
        input.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, l*sizeof(cmn::GPUFLOAT), imdata);
        cl_buffer_region bg{0, sizeof(cmn::GPUFLOAT)*l};
        input.m_reChannels.emplace_back(input.m_reShMem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &bg));
        input.m_imChannels.emplace_back(input.m_imShMem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &bg));

        uut.setInput(input);

        uut.runFwdPropagation(oclContext);
        uut.printFwdProfilingInfo();

        gpu::BufferIO output = uut.getOutput();

        cmn::GPUFLOAT sum = 0;
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_reChannels.front(), sizeof(cmn::GPUFLOAT), &sum);
        ASSERT_EQ(sum, l+1);
        sum = 0;
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_imChannels.front(), sizeof(cmn::GPUFLOAT), &sum);
        ASSERT_EQ(sum, l+1);
    }
}


TEST(FCLayerTests, fclayer_forward_propagation_test_two_neurons)
{
    gpu::OpenCLContext oclContext(envReader);
    gpu::OpenCLProgram oclProgram(envReader);

    const unsigned int inputSize = 3;
    const unsigned int outputSize = 2;

    cmn::GPUFLOAT redata[inputSize];
    cmn::GPUFLOAT imdata[inputSize];
    std::vector<cmn::GPUFLOAT> weights = {1,2,3,0,1,2,4,5,6,3,4,5};
    std::vector<cmn::GPUFLOAT> bias = {1, 1, 1, 1};
    for(unsigned int i = 0; i < inputSize; ++i)
    {
        redata[i] = 1.0;
        imdata[i] = 1.0;
    }
    std::vector<cmn::GPUFLOAT> expRe = {0.9997, 0.9994};
    std::vector<cmn::GPUFLOAT> expIm = {0.0006, -0.0003};

    conf::FCParams params(1, 1, inputSize, outputSize, 0.1, 0, conf::FCParams::WeightsType::COMPLEX, 1, 0, "complextanh", "fclayer_forward_propagation_test_two_neurons");

    if(oclProgram.compile(oclContext))
    {

        net::FCLayer uut(params, weights, bias, envReader, oclProgram, oclContext);

        gpu::BufferIO input;
        input.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, inputSize*sizeof(cmn::GPUFLOAT), redata);
        input.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, inputSize*sizeof(cmn::GPUFLOAT), imdata);
        cl_buffer_region bg{0, sizeof(cmn::GPUFLOAT)*inputSize};
        input.m_reChannels.emplace_back(input.m_reShMem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &bg));
        input.m_imChannels.emplace_back(input.m_imShMem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &bg));

        uut.setInput(input);

        uut.runFwdPropagation(oclContext);
        uut.printFwdProfilingInfo();

        gpu::BufferIO output = uut.getOutput();

        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_reShMem, outputSize*sizeof(cmn::GPUFLOAT), redata);
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_imShMem, outputSize*sizeof(cmn::GPUFLOAT), imdata);
        for(size_t i = 0; i < outputSize; ++i)
        {
            EXPECT_NEAR(expRe[i], redata[i], 0.0001);
            EXPECT_NEAR(expIm[i], imdata[i], 0.0001);
        }
    }
}

TEST(FCLayerTests, fclayer_forward_and_back_propagation_test_two_neurons_in_one_neuron_out)
{
    gpu::OpenCLContext oclContext(envReader);
    gpu::OpenCLProgram oclProgram(envReader);

    const unsigned int inputSize = 3;
    const unsigned int outputSize = 2;
    const unsigned int outputSize2nd = 1;

    cmn::GPUFLOAT redata[inputSize*outputSize*sizeof(cmn::GPUFLOAT)+4];
    cmn::GPUFLOAT imdata[inputSize*outputSize*sizeof(cmn::GPUFLOAT)+4];
    std::vector<cmn::GPUFLOAT> weights = {0.1,0.2,0.3,0.0,0.1,0.2,0.4,0.5,0.6,0.3,0.4,0.5};
    std::vector<cmn::GPUFLOAT> weights2nd = {0.1,0.2,0.3,0.4}; //0.1+j0.3 , 0.2+j0.4
    std::vector<cmn::GPUFLOAT> bias = {1, 1, 1, 1};
    std::vector<cmn::GPUFLOAT> bias2nd = {1, 1};
    for(unsigned int i = 0; i < inputSize; ++i)
    {
        redata[i] = 1.0;
        imdata[i] = 0.0;
    }

    conf::FCParams params(1, 1, inputSize, outputSize, 0.1, 0, conf::FCParams::WeightsType::COMPLEX, 1, 0, "complextanh", "fclayer_forward_propagation_test_two_neurons");
    conf::FCParams params2nd(1, 1, outputSize, outputSize2nd, 0.1, 0, conf::FCParams::WeightsType::COMPLEX, 1, 0, "complextanh", "fclayer_forward_propagation_test_two_neurons_2nd");

    if(oclProgram.compile(oclContext))
    {

        net::FCLayer uut(params, weights, bias, envReader, oclProgram, oclContext);
        net::FCLayer uut2nd(params2nd, weights2nd, bias2nd, envReader, oclProgram, oclContext);

        gpu::BufferIO input;
        input.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, 100*sizeof(cmn::GPUFLOAT));
        input.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, 100*sizeof(cmn::GPUFLOAT));
        input.m_reShMemBkp = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, 100*sizeof(cmn::GPUFLOAT));
        input.m_imShMemBkp = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, 100*sizeof(cmn::GPUFLOAT));

        uut.writeToBuffer(oclContext.getCommandQueue(), input.m_reShMem, inputSize*sizeof(cmn::GPUFLOAT), redata);
        uut.writeToBuffer(oclContext.getCommandQueue(), input.m_imShMem, inputSize*sizeof(cmn::GPUFLOAT), imdata);

        cl_buffer_region bg{0, sizeof(cmn::GPUFLOAT)*inputSize};
        input.m_reChannels.emplace_back(input.m_reShMem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &bg));
        input.m_imChannels.emplace_back(input.m_imShMem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &bg));
        cmn::GPUFLOAT desired = 0;
        input.m_reDesired = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cmn::GPUFLOAT), &desired);
        input.m_imDesired = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cmn::GPUFLOAT), &desired);

        conf::ErrorCalculationLayerParams errCalcParams(1, "meansquare", "fclayertest");
        net::ErrorCalculationLayer err1(errCalcParams, envReader, oclProgram, oclContext);

        uut.setInput(input);
        uut >> uut2nd >> err1;


        err1.runFwdPropagation(oclContext);
        err1.printFwdProfilingInfo();

        std::vector<cmn::GPUFLOAT> expRe = {1.1512};
        std::vector<cmn::GPUFLOAT> expIm = {-0.0522};

        gpu::BufferIO output = err1.getOutput();

        err1.readFromBuffer(oclContext.getCommandQueue(), output.m_reShMem, outputSize2nd*sizeof(cmn::GPUFLOAT), redata);
        err1.readFromBuffer(oclContext.getCommandQueue(), output.m_imShMem, outputSize2nd*sizeof(cmn::GPUFLOAT), imdata);
        for(size_t i = 0; i < outputSize2nd; ++i)
        {
            EXPECT_NEAR(expRe[i], desired - redata[i], 0.0001);
            EXPECT_NEAR(expIm[i], desired - imdata[i], 0.0001);
        }

        expRe.clear();
        expIm.clear();
        expRe.push_back(1.0714);
        expRe.push_back(1.0041);
        expIm.push_back(0.0451);
        expIm.push_back(-0.0129);
        LayerInput l2input = uut2nd.getLayerInput();
        uut2nd.readFromBuffer(oclContext.getCommandQueue(), l2input.m_re, outputSize*sizeof(cmn::GPUFLOAT), redata);
        uut2nd.readFromBuffer(oclContext.getCommandQueue(), l2input.m_im, outputSize*sizeof(cmn::GPUFLOAT), imdata);
        for(size_t i = 0; i < outputSize; ++i)
        {
            EXPECT_NEAR(expRe[i], redata[i], 0.0001);
            EXPECT_NEAR(expIm[i], imdata[i], 0.0001);
        }


        uut << uut2nd << err1;
        uut.runBckPropagation(oclContext);
        uut.printBckProfilingInfo();

        expRe.clear();
        expIm.clear();
        expRe.push_back(0.5100);
        expRe.push_back(0.5775);
        expIm.push_back(0.4133);
        expIm.push_back(0.5270);
        LayerWeights newWeights = uut2nd.getWeights();
        uut.readFromBuffer(oclContext.getCommandQueue(), newWeights[0].m_re, newWeights[0].m_size*sizeof(cmn::GPUFLOAT), redata);
        uut.readFromBuffer(oclContext.getCommandQueue(), newWeights[0].m_im, newWeights[0].m_size*sizeof(cmn::GPUFLOAT), imdata);
        for(size_t i = 0; i < newWeights[0].m_size; ++i)
        {
            EXPECT_NEAR(expRe[i], redata[i], 0.0001);
            EXPECT_NEAR(expIm[i], imdata[i], 0.0001);
        }

        expRe = {-0.0025, -0.0010, 0.0004};
        expIm = {0.0026, 0.0051, 0.0076};
        output = uut.getBkpOutput();
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_reShMem, inputSize*sizeof(cmn::GPUFLOAT), redata);
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_imShMem, inputSize*sizeof(cmn::GPUFLOAT), imdata);
        for(size_t i = 0; i < inputSize; ++i)
        {
            EXPECT_NEAR(expRe[i], redata[i], 0.0001);
            EXPECT_NEAR(expIm[i], imdata[i], 0.0001);
        }


        expRe = {0.0989, 0.1989, 0.2989, 0.3957, 0.4957, 0.5957};
        expIm = {0.0219, 0.1219, 0.2219, 0.2978, 0.3978, 0.4978};
        newWeights = uut.getWeights();
        for(size_t nid = 0; nid < newWeights.size(); ++nid)
        {
            uut.readFromBuffer(oclContext.getCommandQueue(), newWeights[nid].m_re, newWeights[nid].m_size*sizeof(cmn::GPUFLOAT), redata);
            uut.readFromBuffer(oclContext.getCommandQueue(), newWeights[nid].m_im, newWeights[nid].m_size*sizeof(cmn::GPUFLOAT), imdata);
            for(size_t i = 0; i < newWeights[nid].m_size; ++i)
            {
                EXPECT_NEAR(expRe[nid*newWeights[nid].m_size + i], redata[i], 0.0001);
                EXPECT_NEAR(expIm[nid*newWeights[nid].m_size + i], imdata[i], 0.0001);
            }
        }
    }
}
