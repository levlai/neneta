#pragma once

#include <OpenCLExecutionPlan.h>
#include <IPersistedLayer.h>
#include <ConvLayerConfiguration.h>
#include <Persistance.h>
#include <IOpenCLChainableExecutionPlan.h>
#include <KernelConfiguration.h>
#include <OpenCLKernel.h>
#include <Utils.h>
#include <Types.h>

namespace neneta
{
namespace net
{

class ConvLayer : public gpu::OpenCLExecutionPlan, public IPersistedLayer, public gpu::IOpenCLChainableExecutionPlan
{
public:    
    ConvLayer(const std::string& layerId, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext);
    ConvLayer(const conf::ConvLayerParams& params, const std::vector<cmn::GPUFLOAT>& weights, const std::vector<cmn::GPUFLOAT>& bias,
              const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext);
    ~ConvLayer();

    //interface impl
    void store() override;
    void restore() override;

    // IOpenCLChainableExecutionPlan interface    
    void setInput(gpu::BufferIO input) override;
    gpu::BufferIO getOutput() override;
    void setBkpInput(gpu::BufferIO input) override;
    gpu::BufferIO getBkpOutput() override;    

    //helpers for bp
    void prepareBuffers(gpu::BufferIO input);
    void calculateDeltas();
    void rotateInput();    
    void calculateErrors(const unsigned int outKernel);
    void updateWeights(const unsigned int outKernel);

    //getters
    const LayerInput& getLayerInput() const { return m_layerInput; }
    const LayerWeights& getWeights() const { return m_weights; }
    const LayerBiases& getBiases() const { return m_biases; }
    const LayerDeltas& getDeltas() const { return m_deltas; }

private:
    void init(std::vector<cmn::GPUFLOAT>& rawdata);
    void planForwardExecution();
    void planBackwardExecution();
    void readWeightsAndBiases(std::vector<cmn::GPUFLOAT>& weightsandbiases);

private:    
    pers::Persistance m_persistance;
    conf::ConvLayerParams m_layerParameters;
    gpu::OpenCLKernelParameters m_oclKernelParameters;
    unsigned int m_inputChannelSize;
    unsigned int m_outputWidth;
    unsigned int m_outputChannelSize;    
    const gpu::OpenCLContext& m_clContext;    
    const gpu::OpenCLProgram& m_clProgram;
    unsigned int m_maxWGSize;
    gpu::BufferIO m_io;
    LayerInput m_layerInput;
    LayerWeights m_weights;
    LayerBiases m_biases;
    LayerDeltas m_deltas;
    unsigned int m_weightsSizePerChannel;

};

} // net
} // neneta
