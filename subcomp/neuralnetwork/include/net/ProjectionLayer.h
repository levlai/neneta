#pragma once

#include <OpenCLExecutionPlan.h>
#include <IOpenCLChainableExecutionPlan.h>
#include <ProjectionLayerConfiguration.h>
#include "Utils.h"

namespace neneta
{

namespace gpu
{
class OpenCLContext;
}

namespace net
{

class ProjectionLayer : public gpu::OpenCLExecutionPlan, public gpu::IOpenCLChainableExecutionPlan
{
public:
    ProjectionLayer(const std::string& layerId, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext);
    ProjectionLayer(const conf::ProjectionLayerParams& params, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext);
    ~ProjectionLayer();

    // IOpenCLChainableExecutionPlan interface    
    void setInput(gpu::BufferIO input) override;
    gpu::BufferIO getOutput() override;
    void setBkpInput(gpu::BufferIO input) override;
    gpu::BufferIO getBkpOutput() override;        

private:
    void prepareBuffers(gpu::BufferIO input);    
    void calculateDeltas();
    void calculateErrors();

private:    
    const gpu::OpenCLContext& m_clContext;
    unsigned int m_numOfNeuronsInLayer;
    conf::ProjectionLayerParams m_layerParameters;
    LayerWeights m_weights;
    LayerDeltas m_deltas;
    gpu::BufferIO m_io;
};

} // net
} // neneta
