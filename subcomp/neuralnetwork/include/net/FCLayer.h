#pragma once

#include <OpenCLExecutionPlan.h>
#include <IOpenCLChainableExecutionPlan.h>
#include <FCLayerConfiguration.h>
#include <IPersistedLayer.h>
#include <Persistance.h>
#include <Utils.h>

namespace neneta
{
namespace net
{

class FCLayer : public gpu::OpenCLExecutionPlan, public IPersistedLayer, public gpu::IOpenCLChainableExecutionPlan
{
public:
    FCLayer(const std::string& layerId, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext);
    FCLayer(const conf::FCParams& params, const std::vector<cmn::GPUFLOAT>& weights, const std::vector<cmn::GPUFLOAT>& bias,
            const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext);
    ~FCLayer();

    // IOpenCLChainableExecutionPlan interface    
    void setInput(gpu::BufferIO input) override;
    gpu::BufferIO getOutput() override;
    void setBkpInput(gpu::BufferIO input) override;
    gpu::BufferIO getBkpOutput() override;    

    // IPersistedLayer interface
    void store() override;
    void restore() override;

    //getters
    const LayerInput& getLayerInput() const { return m_layerInput; }
    const LayerWeights& getWeights() const { return m_weights; }
    const LayerBiases& getBiases() const { return m_biases; }
    const LayerDeltas& getDeltas() const { return m_deltas; }

private:
    void prepareBuffers(gpu::BufferIO input);
    void init(std::vector<cmn::GPUFLOAT>& rawdata);
    void calculateDeltas();
    void calculateErrors();
    void updateWeights();
    void readWeightsAndBiases(std::vector<cmn::GPUFLOAT>& weightsandbiases);
private:    
    pers::Persistance m_persistance;
    conf::FCParams m_layerParameters;
    unsigned int m_channelVectorSize;
    const gpu::OpenCLContext& m_clContext;    
    const gpu::OpenCLProgram& m_clProgram;
    unsigned int m_maxWGSize;    
    gpu::BufferIO m_io;
    LayerWeights m_weights;
    LayerInput m_layerInput;
    LayerDeltas m_deltas;
    LayerBiases m_biases;
    cl::Buffer m_reResult;
    cl::Buffer m_imResult;
    cl::Buffer m_reSmallTemp;
    cl::Buffer m_imSmallTemp;
};

} // net
} // neneta
