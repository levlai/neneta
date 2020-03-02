#pragma once

#include <OpenCLExecutionPlan.h>
#include <IPersistedLayer.h>
#include <FFTConvLayerConfiguration.h>
#include <Persistance.h>
#include <IOpenCLChainableExecutionPlan.h>
#include <FourierTransform.h>
#include <KernelConfiguration.h>
#include <OpenCLKernel.h>

namespace neneta
{
namespace net
{

class FFTConvLayer : public gpu::OpenCLExecutionPlan, public IPersistedLayer, public gpu::IOpenCLChainableExecutionPlan
{
public:
    FFTConvLayer(const std::string& layerId, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext);
    ~FFTConvLayer();

    //interface impl
    void store() override;
    void restore() override;

    // IOpenCLChainableExecutionPlan interface    
    void setInput(gpu::BufferIO input) override;
    gpu::BufferIO getOutput() override;
    void setBkpInput(gpu::BufferIO input) override;
    gpu::BufferIO getBkpOutput() override;    

private:
    void init(const conf::ConfigurationReader& confReader);
    void prepareBuffers(gpu::BufferIO input);

private:    
    pers::Persistance m_persistance;
    conf::FFTConvLayerParams m_layerParameters;
    gpu::OpenCLKernelParameters m_oclKernelParameters;
    const gpu::OpenCLContext& m_clContext;    
    const gpu::OpenCLProgram& m_clProgram;
    std::vector<cmn::GPUFLOAT> m_kernels;
    std::vector<std::vector<FourierTransform>> m_fftKernels;
    gpu::BufferIO m_io;
    gpu::BufferIO m_tempBuffers;
    cl::Buffer m_reTmp;
    cl::Buffer m_imTmp;    

};

} // net
} // neneta
