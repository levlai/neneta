#pragma once

#include <OpenCLExecutionPlan.h>
#include <IOpenCLChainableExecutionPlan.h>
#include <ErrorCalculationLayerConfiguration.h>
#include <IOpenCLOutputExecutionPlan.h>
#include "Utils.h"

namespace neneta
{

namespace gpu
{
class OpenCLContext;
}

namespace net
{

class ErrorCalculationLayer : public gpu::OpenCLExecutionPlan, public gpu::IOpenCLChainableExecutionPlan, public gpu::IOpenCLOutputExecutionPlan
{
public:
    ErrorCalculationLayer(const std::string& layerId, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext);
    ErrorCalculationLayer(const conf::ErrorCalculationLayerParams& params, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext);
    ~ErrorCalculationLayer();

    // IOpenCLChainableExecutionPlan interface    
    void setInput(gpu::BufferIO input) override;
    gpu::BufferIO getOutput() override;
    void setBkpInput(gpu::BufferIO input) override;
    gpu::BufferIO getBkpOutput() override;        

    // IOpenCLOutputExecutionPlan interface
    cmn::GPUFLOAT getLoss() const;
    cmn::GPUFLOAT getAccuracy() const;

private:
    void prepareBuffers(gpu::BufferIO input);
    void calculateDeltas();
    void calculateLossVector();
    void calculateSumOfLossVector();
    void calculateAccuracy();

private:    
    const gpu::OpenCLContext& m_clContext;
    unsigned int m_maxWGSize;
    conf::ErrorCalculationLayerParams m_layerParameters;    
    LayerDeltas m_deltas;
    gpu::BufferIO m_io;
    cl::Buffer m_loss;
    cl::Buffer m_accuracy;
    cl::Buffer m_smallTemp;
    std::vector<cmn::GPUFLOAT> m_errors;
};

} // net
} // neneta
