#pragma once

#include <OpenCLExecutionPlan.h>
#include <IOpenCLChainableExecutionPlan.h>
#include <InputLayerConfiguration.h>
#include <IOpenCLInputExecutionPlan.h>

namespace neneta
{

namespace gpu
{
class OpenCLContext;
}

namespace net
{

class InputLayer : public gpu::OpenCLExecutionPlan, public gpu::IOpenCLChainableExecutionPlan, public gpu::IOpenCLInputExecutionPlan
{
public:
    InputLayer(const std::string& layerId, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext);
    InputLayer(const conf::InputLayerParams& params, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext);
    ~InputLayer();

    // IOpenCLChainableExecutionPlan interface    
    void setInput(gpu::BufferIO input) override;
    gpu::BufferIO getOutput() override;
    void setBkpInput(gpu::BufferIO input) override;
    gpu::BufferIO getBkpOutput() override;

    // IOpenCLInputExecutionPlan interface
    void setInput(ImageType &image);

private:    
    const gpu::OpenCLContext& m_clContext;
    conf::InputLayerParams m_layerParameters;
    gpu::BufferIO m_sharedGPUMemory;
};

} // net
} // neneta
