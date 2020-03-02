#pragma once

#include <OpenCLExecutionPlan.h>
#include <IOpenCLChainableExecutionPlan.h>
#include <SoftMaxLayerConfiguration.h>
#include <IPersistedLayer.h>
#include "Utils.h"
#include <Persistance.h>

namespace neneta
{
namespace net
{

class SoftMaxLayer : public gpu::OpenCLExecutionPlan,  public IPersistedLayer, public gpu::IOpenCLChainableExecutionPlan
{
public:
    SoftMaxLayer(const std::string& layerId, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext);
    SoftMaxLayer(const conf::SoftMaxParams& layerId, const std::vector<cmn::GPUFLOAT>& weights, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext);
    ~SoftMaxLayer();

    // IOpenCLChainableExecutionPlan interface    
    void setInput(gpu::BufferIO input) override;
    gpu::BufferIO getOutput() override;
    void setBkpInput(gpu::BufferIO input) override;
    gpu::BufferIO getBkpOutput() override;        

    // IPersistedLayer interface
    void store() override;
    void restore() override;

private:
    void prepareBuffers(gpu::BufferIO input);
    void init(std::vector<cmn::GPUFLOAT>& rawdata);
    void calculateDeltas();
    void calculateErrors();
    void updateWeights();
    void readWeightsAndBiases(std::vector<cmn::GPUFLOAT>& weightsandbiases);

private:    
    pers::Persistance m_persistance;
    conf::SoftMaxParams m_layerParameters;    
    unsigned int m_channelVectorSize;
    unsigned int m_numOfNeuronInputs;    
    const gpu::OpenCLContext& m_clContext;    
    const gpu::OpenCLProgram& m_clProgram;    
    unsigned int m_maxWGSize;
    unsigned int m_numOfWGPerNeuron;
    LayerWeights m_weights;
    LayerBiases m_biases;
    LayerDeltas m_deltas;
    LayerInput m_layerInput;
    cl::Buffer m_regularizationTerm;    
    cl::Buffer m_maxValue;
    gpu::BufferIO m_io;
    cl::Buffer m_reResult;
    cl::Buffer m_imResult;
    cl::Buffer m_reSmallTemp;
    cl::Buffer m_imSmallTemp;
};

} // net
} // neneta
