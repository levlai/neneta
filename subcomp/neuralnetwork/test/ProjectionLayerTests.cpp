#include <gtest/gtest.h>
#include <ProjectionLayer.h>
#include <OpenCLContext.h>
#include <OpenCLProgram.h>
#include <OpenCLExecutionPlan.h>


using namespace neneta;
using namespace neneta::net;


extern conf::ConfigurationReader envReader;

TEST(ProjectionLayerTests, projection_layer_forward_and_back_propagation_test)
{
    gpu::OpenCLContext oclContext(envReader);
    gpu::OpenCLProgram oclProgram(envReader);

    const unsigned int channels = 10;

    cmn::GPUFLOAT redata[channels];
    cmn::GPUFLOAT imdata[channels];
    cmn::GPUFLOAT abs[channels];
    for(unsigned int i = 0; i < channels; ++i)
    {
        redata[i] = i;
        imdata[i] = i+1;
        abs[i] = std::pow(i,2) + std::pow(i+1,2);
    }

    cmn::GPUFLOAT redataBkp[channels];
    cmn::GPUFLOAT imdataBkp[channels];
    cmn::GPUFLOAT bkpResultRe[channels];
    cmn::GPUFLOAT bkpResultIm[channels];
    for(unsigned int i = 0; i < channels; ++i)
    {
        redataBkp[i] = i;
        imdataBkp[i] = 0;
        bkpResultRe[i] = 2*redata[i]*redataBkp[i];
        bkpResultIm[i] = -2*imdata[i]*redataBkp[i];
    }

    conf::ProjectionLayerParams params(channels, "absolute", "ProjectionTest");

    if(oclProgram.compile(oclContext))
    {

        net::ProjectionLayer uut(params, envReader, oclProgram, oclContext);

        gpu::BufferIO input;
        input.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, channels*sizeof(cmn::GPUFLOAT), redata);
        input.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, channels*sizeof(cmn::GPUFLOAT), imdata);

        gpu::BufferIO bkpInput;
        bkpInput.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, channels*sizeof(cmn::GPUFLOAT), redataBkp);
        bkpInput.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, channels*sizeof(cmn::GPUFLOAT), imdataBkp);

        //run fp
        uut.setInput(input);
        uut.runFwdPropagation(oclContext);
        uut.printFwdProfilingInfo();

        //read outputs and check
        gpu::BufferIO output = uut.getOutput();
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_reShMem, sizeof(cmn::GPUFLOAT)*channels, redata);
        uut.readFromBuffer(oclContext.getCommandQueue(), output.m_imShMem, sizeof(cmn::GPUFLOAT)*channels, imdata);
        for(unsigned int i = 0; i < channels; ++i)
        {
            ASSERT_FLOAT_EQ(abs[i], (cmn::GPUFLOAT)redata[i]);
            ASSERT_FLOAT_EQ(0, (cmn::GPUFLOAT)imdata[i]);
        }

        //run bp
        uut.setBkpInput(bkpInput);
        uut.runBckPropagation(oclContext);
        uut.printBckProfilingInfo();
        gpu::BufferIO bkpOutput = uut.getBkpOutput();

        uut.readFromBuffer(oclContext.getCommandQueue(), bkpOutput.m_reShMem, sizeof(cmn::GPUFLOAT)*channels, redataBkp);
        uut.readFromBuffer(oclContext.getCommandQueue(), bkpOutput.m_imShMem, sizeof(cmn::GPUFLOAT)*channels, imdataBkp);
        for(unsigned int i = 0; i < channels; ++i)
        {
            ASSERT_FLOAT_EQ(bkpResultRe[i], (cmn::GPUFLOAT)redataBkp[i]);
            ASSERT_FLOAT_EQ(bkpResultIm[i], (cmn::GPUFLOAT)imdataBkp[i]);            
        }
    }
}
