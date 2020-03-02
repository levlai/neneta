#pragma once

#include <OpenCLExecutionPlan.h>
#include <IOpenCLChainableExecutionPlan.h>
#include <PoolingLayerConfiguration.h>
#include <Types.h>

namespace neneta
{
namespace net
{

class SpectralPoolingLayer : public gpu::OpenCLExecutionPlan, public gpu::IOpenCLChainableExecutionPlan
{
public:
    SpectralPoolingLayer(const std::string& layerId, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext);
    ~SpectralPoolingLayer();

    void getRe(std::vector<std::vector<cmn::GPUFLOAT>>& re);
    void getIm(std::vector<std::vector<cmn::GPUFLOAT>>& im);

    // IOpenCLChainableExecutionPlan interface    
    void setInput(gpu::BufferIO input);
    gpu::BufferIO getOutput();
    void setBkpInput(gpu::BufferIO input) override;
    gpu::BufferIO getBkpOutput() override;        

private:
    void init(const conf::ConfigurationReader& confReader);
    void prepareBuffers(gpu::BufferIO input);

private:    
    conf::SpectralPoolingParams m_layerParameters;
    gpu::OpenCLKernelParameters m_oclKernelParameters;
    const gpu::OpenCLContext& m_clContext;    
    const gpu::OpenCLProgram& m_clProgram;
    gpu::BufferIO m_io;
    gpu::BufferIO m_tempBuffers;
    cl::Buffer m_reTmp;
    cl::Buffer m_imTmp;
};

} // net
} // neneta
