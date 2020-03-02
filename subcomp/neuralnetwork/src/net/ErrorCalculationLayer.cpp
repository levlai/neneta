#include <ErrorCalculationLayer.h>
#include <OpenCLContext.h>
#include <Utils.h>

using namespace neneta;
using namespace neneta::net;


ErrorCalculationLayer::ErrorCalculationLayer(const std::string& layerId, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext)
    : gpu::OpenCLExecutionPlan(layerId, confReader, program)
    , m_clContext(oclContext)
    , m_maxWGSize(m_clContext.getDevice().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>())
    , m_layerParameters(conf::ErrorCalculationLayerConfiguration(getId(), confReader).getParamSet(getId()))                                               
    , m_deltas(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*sizeof(cmn::GPUFLOAT)),
               cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*sizeof(cmn::GPUFLOAT)),
               m_layerParameters.m_channels)
    , m_loss(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*sizeof(cmn::GPUFLOAT))
    , m_accuracy(m_clContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT))
    , m_smallTemp(m_clContext.getContext(), CL_MEM_READ_WRITE,
                  sizeof(cmn::GPUFLOAT)*(m_layerParameters.m_channels/m_maxWGSize + (m_layerParameters.m_channels%m_maxWGSize != 0)))
{    
    initBuffers<cmn::GPUFLOAT>(*this, m_clContext.getCommandQueue(), m_deltas.m_size, m_deltas.m_re, 1, m_deltas.m_im, 0);
}

ErrorCalculationLayer::ErrorCalculationLayer(const conf::ErrorCalculationLayerParams& params, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram &program, const gpu::OpenCLContext &oclContext)
    : gpu::OpenCLExecutionPlan("errcalclayer", confReader, program)
    , m_clContext(oclContext)
    , m_maxWGSize(m_clContext.getDevice().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>())
    , m_layerParameters(params)
    , m_deltas(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*sizeof(cmn::GPUFLOAT)),
               cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*sizeof(cmn::GPUFLOAT)),
               m_layerParameters.m_channels)
    , m_loss(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*sizeof(cmn::GPUFLOAT))
    , m_accuracy(m_clContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT))
    , m_smallTemp(m_clContext.getContext(), CL_MEM_READ_WRITE,
                  sizeof(cmn::GPUFLOAT)*(m_layerParameters.m_channels/m_maxWGSize + (m_layerParameters.m_channels%m_maxWGSize != 0)))
{        
    initBuffers<cmn::GPUFLOAT>(*this, m_clContext.getCommandQueue(), m_deltas.m_size, m_deltas.m_re, 1, m_deltas.m_im, 0);
}

ErrorCalculationLayer::~ErrorCalculationLayer()
{
}

void ErrorCalculationLayer::setInput(gpu::BufferIO input)
{    
    prepareBuffers(input);
    calculateLossVector();
    calculateSumOfLossVector();
    calculateAccuracy();
    calculateDeltas();
}

void ErrorCalculationLayer::calculateAccuracy()
{
    gpu::OpenCLKernelParameters kernel(1,1);
    planFwd(m_layerParameters.m_errorFunc+std::string("_acc"), kernel, m_io.m_reDesired, m_io.m_imDesired,
                                                           m_io.m_reShMem, m_io.m_imShMem,
                                                           m_layerParameters.m_channels, m_accuracy);
}

void ErrorCalculationLayer::calculateDeltas()
{
    gpu::OpenCLKernelParameters kernel(m_layerParameters.m_channels, 0);
    planFwd(m_layerParameters.m_errorFunc, kernel,
            m_deltas.m_re, m_deltas.m_im,
            m_io.m_reDesired, m_io.m_imDesired,
            m_io.m_reShMem, m_io.m_imShMem);

    planCopyBufferFwd(m_deltas.m_re, m_io.m_reShMem, m_deltas.m_size*sizeof(cmn::GPUFLOAT));
    planCopyBufferFwd(m_deltas.m_im, m_io.m_imShMem, m_deltas.m_size*sizeof(cmn::GPUFLOAT));
}

void ErrorCalculationLayer::calculateLossVector()
{
    gpu::OpenCLKernelParameters vhpKernel(m_layerParameters.m_channels, 0);
    planFwd(m_layerParameters.m_errorFunc+std::string("_act"),
            vhpKernel,
            m_loss,
            m_io.m_reDesired, m_io.m_imDesired,
            m_io.m_reShMem, m_io.m_imShMem);
}

void ErrorCalculationLayer::calculateSumOfLossVector()
{
    unsigned int inputSize = m_layerParameters.m_channels;
    do
    {
        gpu::OpenCLKernelParameters kparams = calculateKernelParameters(inputSize, m_maxWGSize);
        planFwd("loss_sum",
                kparams,
                m_smallTemp, m_loss,
                inputSize, m_layerParameters.m_channels,
                cl::Local(sizeof(cmn::GPUFLOAT)*m_maxWGSize));

        inputSize = inputSize/m_maxWGSize + (inputSize%m_maxWGSize != 0);

        if(inputSize > 1)
        {
            planCopyBufferFwd(m_smallTemp, m_loss, inputSize*sizeof(cmn::GPUFLOAT));
        }
        else
        {
            planCopyBufferFwd(m_smallTemp, m_loss, sizeof(cmn::GPUFLOAT));
        }
    } while(inputSize > 1);
}

gpu::BufferIO ErrorCalculationLayer::getOutput()
{
    return m_io;
}

void ErrorCalculationLayer::setBkpInput(gpu::BufferIO)
{
}

gpu::BufferIO ErrorCalculationLayer::getBkpOutput()
{
    return m_io;
}

void ErrorCalculationLayer::prepareBuffers(gpu::BufferIO input)
{
    m_io = input;
}


cmn::GPUFLOAT neneta::net::ErrorCalculationLayer::getLoss() const
{
    cmn::GPUFLOAT data = 0.0;
    readFromBuffer(m_clContext.getCommandQueue(), m_loss, sizeof(cmn::GPUFLOAT), const_cast<cmn::GPUFLOAT*>(&data));
    return data;
}

cmn::GPUFLOAT neneta::net::ErrorCalculationLayer::getAccuracy() const
{
    cmn::GPUFLOAT data = 0.0;
    readFromBuffer(m_clContext.getCommandQueue(), m_accuracy, sizeof(cmn::GPUFLOAT), const_cast<cmn::GPUFLOAT*>(&data));
    return data;
}
