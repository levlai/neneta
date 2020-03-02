#pragma once

#include <OpenCLExecutionPlan.h>
#include <Image.h>
#include <IOpenCLChainableExecutionPlan.h>
#include <FourierConfiguration.h>

namespace neneta
{

namespace net
{

class FourierTransform : public gpu::OpenCLExecutionPlan, public gpu::IOpenCLChainableExecutionPlan
{
public:
    FourierTransform(const std::string& layerId, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext);
    FourierTransform(const conf::FourierParams& params, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext);
    FourierTransform(const FourierTransform& cp);
    FourierTransform& operator=(const FourierTransform& cp) = delete;
    ~FourierTransform();

    void getRe(std::vector<std::vector<cmn::GPUFLOAT>>& re);
    void getIm(std::vector<std::vector<cmn::GPUFLOAT>>& im);

    // IChainableLayer interface    
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
