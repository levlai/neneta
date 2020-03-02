#pragma once

#include <FourierConfiguration.h>
#include <OpenCLExecutionPlan.h>
#include <IOpenCLChainableExecutionPlan.h>
#include <Image.h>

namespace neneta
{

namespace net
{

class InverseFourierTransform : public gpu::OpenCLExecutionPlan, public gpu::IOpenCLChainableExecutionPlan
{
public:
    InverseFourierTransform(const std::string& layerId, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext);
    ~InverseFourierTransform();

    void getRe(std::vector<std::vector<cmn::GPUFLOAT>>& re);
    void getIm(std::vector<std::vector<cmn::GPUFLOAT>>& im);

    // IOpenCLChainableExecutionPlan interface    
    void setInput(gpu::BufferIO input) override;
    gpu::BufferIO getOutput() override;
    void setBkpInput(gpu::BufferIO input) override;
    gpu::BufferIO getBkpOutput() override;        

private:
    void init();

private:
    conf::FourierParams m_layerParameters;
    gpu::OpenCLKernelParameters m_oclKernelParameters;
    const gpu::OpenCLContext& m_clContext;
    gpu::BufferIO m_io;
    cl::Buffer m_reTmp;
    cl::Buffer m_imTmp;   
};

} // net
} // neneta
