#include <ProjectionLayer.h>
#include <OpenCLContext.h>
#include <Utils.h>

using namespace neneta;
using namespace neneta::net;


ProjectionLayer::ProjectionLayer(const std::string& layerId, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext)
    : gpu::OpenCLExecutionPlan(layerId, confReader, program)
    , m_clContext(oclContext)    
    , m_numOfNeuronsInLayer(1)
    , m_layerParameters(conf::ProjectionLayerConfiguration(getId(), confReader).getParamSet(getId()))
    , m_weights(m_numOfNeuronsInLayer, {cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*sizeof(cmn::GPUFLOAT)),
                                        cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*sizeof(cmn::GPUFLOAT)),
                                        m_layerParameters.m_channels})
    , m_deltas(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*sizeof(cmn::GPUFLOAT2)),
               cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*sizeof(cmn::GPUFLOAT2)),
               m_layerParameters.m_channels)
{        
    initBuffers<cmn::GPUFLOAT>(*this, m_clContext.getCommandQueue(), m_layerParameters.m_channels, m_weights.front().m_re, 1, m_weights.front().m_im, 0);
}

ProjectionLayer::ProjectionLayer(const conf::ProjectionLayerParams& params, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram &program, const gpu::OpenCLContext &oclContext)
    : gpu::OpenCLExecutionPlan("errcalclayer", confReader, program)
    , m_clContext(oclContext)    
    , m_numOfNeuronsInLayer(1)
    , m_layerParameters(params)
    , m_weights(m_numOfNeuronsInLayer, {cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*sizeof(cmn::GPUFLOAT)),
                                        cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*sizeof(cmn::GPUFLOAT)),
                                        m_layerParameters.m_channels})
    , m_deltas(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*sizeof(cmn::GPUFLOAT2)),
               cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*sizeof(cmn::GPUFLOAT2)),
               m_layerParameters.m_channels)
{    
    initBuffers<cmn::GPUFLOAT>(*this, m_clContext.getCommandQueue(), m_layerParameters.m_channels, m_weights.front().m_re, 1, m_weights.front().m_im, 0);
}

ProjectionLayer::~ProjectionLayer()
{
}

void ProjectionLayer::setInput(gpu::BufferIO input)
{    
    prepareBuffers(input);

    gpu::OpenCLKernelParameters kernel(m_layerParameters.m_channels, 0);

    //run through activation function and store its gradient
    planFwd(m_layerParameters.m_projectionFunc.c_str(), kernel,
            m_io.m_reShMem, m_io.m_imShMem,
            m_deltas.m_re, m_deltas.m_im);

}

gpu::BufferIO ProjectionLayer::getOutput()
{
    return m_io;
}

void ProjectionLayer::calculateDeltas()
{
    calculateDeltasFC(*this, m_io, m_deltas);
}

void ProjectionLayer::calculateErrors()
{
    calculateErrorsPL(*this, m_io, m_weights, m_deltas, m_numOfNeuronsInLayer, m_layerParameters.m_channels);
}

void ProjectionLayer::setBkpInput(gpu::BufferIO input)
{
    prepareBuffers(input);
    calculateDeltas();
    calculateErrors();
}

gpu::BufferIO ProjectionLayer::getBkpOutput()
{
    return m_io;
}

void ProjectionLayer::prepareBuffers(gpu::BufferIO input)
{
    m_io = input;
}
