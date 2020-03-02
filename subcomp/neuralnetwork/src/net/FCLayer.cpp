#include <FCLayer.h>
#include <OpenCLContext.h>
#include "Utils.h"

using namespace neneta;
using namespace neneta::net;



FCLayer::FCLayer(const std::string& layerId, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext)
    : gpu::OpenCLExecutionPlan(layerId, confReader, program)
    , m_persistance(confReader)
    , m_layerParameters(conf::FCLayerConfiguration(getId(), confReader).getParamSet(getId()))
    , m_channelVectorSize(std::pow(m_layerParameters.m_inputSize, m_layerParameters.m_inputDim))
    , m_clContext(oclContext)
    , m_clProgram(program)    
    , m_maxWGSize(m_clContext.getDevice().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>())
    , m_layerInput(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*m_channelVectorSize*sizeof(cmn::GPUFLOAT)),
                   cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*m_channelVectorSize*sizeof(cmn::GPUFLOAT)),
                   m_layerParameters.m_channels*m_channelVectorSize)
    , m_deltas(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT2)),
               cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT2)),
               m_layerParameters.m_outputSize)
    , m_biases(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT)),
               cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT)),
               m_layerParameters.m_outputSize)
    , m_reResult(m_clContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT)*m_layerParameters.m_outputSize)
    , m_imResult(m_clContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT)*m_layerParameters.m_outputSize)
    , m_reSmallTemp(m_clContext.getContext(), CL_MEM_READ_WRITE,
                    sizeof(cmn::GPUFLOAT)*((m_channelVectorSize*m_layerParameters.m_channels)/m_maxWGSize + ((m_channelVectorSize*m_layerParameters.m_channels)%m_maxWGSize != 0)))
    , m_imSmallTemp(m_clContext.getContext(), CL_MEM_READ_WRITE,
                sizeof(cmn::GPUFLOAT)*((m_channelVectorSize*m_layerParameters.m_channels)/m_maxWGSize + ((m_channelVectorSize*m_layerParameters.m_channels)%m_maxWGSize != 0)))
{
    restore();
}

FCLayer::FCLayer(const conf::FCParams& layerParams, const std::vector<cmn::GPUFLOAT>& weights, const std::vector<cmn::GPUFLOAT>& bias,
                 const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext)
    : gpu::OpenCLExecutionPlan("fclayer", confReader, program)
    , m_persistance(confReader)
    , m_layerParameters(layerParams)
    , m_channelVectorSize(std::pow(m_layerParameters.m_inputSize, m_layerParameters.m_inputDim))
    , m_clContext(oclContext)
    , m_clProgram(program)    
    , m_maxWGSize(m_clContext.getDevice().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>())
    , m_layerInput(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*m_channelVectorSize*sizeof(cmn::GPUFLOAT)),
                   cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*m_channelVectorSize*sizeof(cmn::GPUFLOAT)),
                   m_layerParameters.m_channels*m_channelVectorSize)
    , m_deltas(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT2)),
               cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT2)),
               m_layerParameters.m_outputSize)
    , m_biases(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT)),
               cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT)),
               m_layerParameters.m_outputSize)
    , m_reResult(m_clContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT)*m_layerParameters.m_outputSize)
    , m_imResult(m_clContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT)*m_layerParameters.m_outputSize)
    , m_reSmallTemp(m_clContext.getContext(), CL_MEM_READ_WRITE, //max num of workgroups
                    sizeof(cmn::GPUFLOAT)*((m_channelVectorSize*m_layerParameters.m_channels)/m_maxWGSize + ((m_channelVectorSize*m_layerParameters.m_channels)%m_maxWGSize != 0)))
    , m_imSmallTemp(m_clContext.getContext(), CL_MEM_READ_WRITE,
                    sizeof(cmn::GPUFLOAT)*((m_channelVectorSize*m_layerParameters.m_channels)/m_maxWGSize + ((m_channelVectorSize*m_layerParameters.m_channels)%m_maxWGSize != 0)))
{
    std::vector<cmn::GPUFLOAT> w = weights;
    w.insert(w.end(), bias.begin(), bias.end());
    init(w);
}

FCLayer::~FCLayer()
{
}

void FCLayer::setInput(gpu::BufferIO input)
{
    prepareBuffers(input);

    //here is m_layerInput reused for tmp storage

    BOOST_LOG_TRIVIAL(debug) << "FCLayer id" << m_layerParameters.m_id << " setting kernels";
    for(unsigned int outChannel = 0; outChannel < m_layerParameters.m_outputSize; ++outChannel)
    { //for each neuron

        unsigned int inputSize = m_channelVectorSize*m_layerParameters.m_channels;
        BOOST_LOG_TRIVIAL(debug) << "FCLayer for neuron " << outChannel << " inputSize = " << inputSize << " calculating product";
        gpu::OpenCLKernelParameters vhpKernel(inputSize, 0);
        planFwd("vec_hadamard_product", vhpKernel, m_layerInput.m_re, m_layerInput.m_im,
                                      m_io.m_reShMem, m_io.m_imShMem,
                                      m_weights[outChannel].m_re, m_weights[outChannel].m_im);

        do
        {
            gpu::OpenCLKernelParameters kparams = calculateKernelParameters(inputSize, m_maxWGSize);
            planFwd("sum", kparams, inputSize, m_reSmallTemp, m_imSmallTemp,
                                 m_layerInput.m_re, m_layerInput.m_im,
                                 cl::Local(sizeof(cmn::GPUFLOAT)*m_maxWGSize),
                                 cl::Local(sizeof(cmn::GPUFLOAT)*m_maxWGSize));

            inputSize = inputSize/m_maxWGSize + (inputSize%m_maxWGSize != 0);

            if(inputSize > 1)
            {
                planCopyBufferFwd(m_reSmallTemp, m_layerInput.m_re, inputSize*sizeof(cmn::GPUFLOAT));
                planCopyBufferFwd(m_imSmallTemp, m_layerInput.m_im, inputSize*sizeof(cmn::GPUFLOAT));
            }
            else
            {
                assert(inputSize == 1 && "inputSize in FCLayer wrong!");
                //run through activation function and store its gradient
                planFwd(m_layerParameters.m_activationFunction.c_str(), gpu::OpenCLKernelParameters(1,1),
                        m_reResult, m_imResult, 1 /*outputOffset*/,
                        m_reSmallTemp, m_imSmallTemp, 0,
                        m_biases.m_re, m_biases.m_im,
                        m_deltas.m_re, m_deltas.m_im,
                        outChannel);
            }
        } while(inputSize > 1);
    }
    //store input
    planCopyBufferFwd(m_io.m_reShMem, m_layerInput.m_re, m_layerParameters.m_channels*m_channelVectorSize*sizeof(cmn::GPUFLOAT));
    planCopyBufferFwd(m_io.m_imShMem, m_layerInput.m_im, m_layerParameters.m_channels*m_channelVectorSize*sizeof(cmn::GPUFLOAT));

    planCopyBufferFwd(m_reResult, m_io.m_reShMem, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT));
    planCopyBufferFwd(m_imResult, m_io.m_imShMem, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT));
}

gpu::BufferIO FCLayer::getOutput()
{
    return m_io;
}

void FCLayer::calculateDeltas()
{
    calculateDeltasFC(*this, m_io, m_deltas);
}

void FCLayer::calculateErrors()
{
    //Calculate errors to be propagated to the left layer
    calculateErrorsFC(*this, m_io, m_weights, m_deltas, m_layerParameters.m_outputSize, m_layerParameters.m_channels*m_channelVectorSize);
}

void FCLayer::updateWeights()
{
    updateWeightsFC(*this, m_weights, m_deltas, m_biases, m_layerInput, m_layerParameters.m_outputSize);
}

void FCLayer::setBkpInput(gpu::BufferIO input)
{
    prepareBuffers(input);
    calculateDeltas();
    calculateErrors();
    updateWeights();
}

gpu::BufferIO FCLayer::getBkpOutput()
{
    return m_io;
}

void FCLayer::store()
{
    std::vector<cmn::GPUFLOAT> weightsandbiases(2*(m_layerParameters.m_channels*m_channelVectorSize+1)*m_layerParameters.m_outputSize);
    readWeightsAndBiases(weightsandbiases);
    m_persistance.storeFloatBlob(getId(), weightsandbiases);
}

void FCLayer::readWeightsAndBiases(std::vector<cmn::GPUFLOAT>& weightsandbiases)
{
    unsigned int offset = 0;
    for(unsigned int i = 0; i < m_layerParameters.m_outputSize; ++i)
    {
        readFromBuffer(m_clContext.getCommandQueue(), m_weights[i].m_re, sizeof(cmn::GPUFLOAT)*m_layerParameters.m_channels*m_channelVectorSize, weightsandbiases.data() + offset);
        offset += m_layerParameters.m_channels*m_channelVectorSize;
        readFromBuffer(m_clContext.getCommandQueue(), m_weights[i].m_im, sizeof(cmn::GPUFLOAT)*m_layerParameters.m_channels*m_channelVectorSize, weightsandbiases.data() + offset);
        offset += m_layerParameters.m_channels*m_channelVectorSize;
    }
    //read biases
    readFromBuffer(m_clContext.getCommandQueue(), m_biases.m_re, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT), weightsandbiases.data() + offset);
    offset += m_layerParameters.m_outputSize;
    readFromBuffer(m_clContext.getCommandQueue(), m_biases.m_im, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT), weightsandbiases.data() + offset);
}

void FCLayer::restore()
{
    // jedan neuron ima brojkanala*2(za re i im)*dim*dim tezina i bias
    // 64*2*16*16
    std::vector<cmn::GPUFLOAT> ungroupedweights;
    m_persistance.restoreFloatBlob(getId(), ungroupedweights);
    if(ungroupedweights.empty())
    {
        BOOST_LOG_TRIVIAL(debug) << "FCLayer generating weights vector";
        for(unsigned int i = 0; i < m_layerParameters.m_outputSize; ++i)
        {
            std::vector<cmn::GPUFLOAT> real = generateRandomVector(m_layerParameters.m_channels*m_channelVectorSize, m_layerParameters.m_weightsMean, m_layerParameters.m_initWeightsDeviation);
            std::vector<cmn::GPUFLOAT> imag(m_layerParameters.m_channels*m_channelVectorSize, 0.0);
            if(m_layerParameters.m_weightsType == conf::FCParams::WeightsType::COMPLEX)
            {
                imag = generateRandomVector(m_layerParameters.m_channels*m_channelVectorSize, m_layerParameters.m_weightsMean, m_layerParameters.m_initWeightsDeviation);
            }
            ungroupedweights.insert(ungroupedweights.end(), real.begin(), real.end());
            ungroupedweights.insert(ungroupedweights.end(), imag.begin(), imag.end());            
        }
        std::vector<cmn::GPUFLOAT> biasReal(m_layerParameters.m_outputSize, m_layerParameters.m_initBias.real());
        std::vector<cmn::GPUFLOAT> biasIm(m_layerParameters.m_outputSize, m_layerParameters.m_initBias.imag());
        ungroupedweights.insert(ungroupedweights.end(), biasReal.begin(), biasReal.end());
        ungroupedweights.insert(ungroupedweights.end(), biasIm.begin(), biasIm.end());
    }
    assert((ungroupedweights.size() == 2*(m_layerParameters.m_channels*m_channelVectorSize+1)*m_layerParameters.m_outputSize) && "FCLayer::restore() doesn't work!");
    init(ungroupedweights);
}

void FCLayer::prepareBuffers(gpu::BufferIO input)
{
    m_io = input;    
}

void FCLayer::init(std::vector<cmn::GPUFLOAT>& rawdata)
{
    //init weights
    for(unsigned int i = 0; i < m_layerParameters.m_outputSize; ++i)
    {
        BOOST_LOG_TRIVIAL(debug) << "FCLayer for neuron " << i << " initializing weights for im and re, num of cmn::GPUFLOATs  " << 2*m_layerParameters.m_channels*m_channelVectorSize ;
        m_weights.emplace_back(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , m_layerParameters.m_channels*m_channelVectorSize*sizeof(cmn::GPUFLOAT), (void*)(rawdata.data() + 2*i*m_layerParameters.m_channels*m_channelVectorSize)),
                               cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , m_layerParameters.m_channels*m_channelVectorSize*sizeof(cmn::GPUFLOAT), (void*)(rawdata.data() + (2*i+1)*m_layerParameters.m_channels*m_channelVectorSize)),
                               m_layerParameters.m_channels*m_channelVectorSize);
    }

    //init biases
    writeToBuffer(m_clContext.getCommandQueue(), m_biases.m_re, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT), rawdata.data() + 2*m_layerParameters.m_outputSize*m_layerParameters.m_channels*m_channelVectorSize);
    writeToBuffer(m_clContext.getCommandQueue(), m_biases.m_im, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT), rawdata.data() + 2*m_layerParameters.m_outputSize*m_layerParameters.m_channels*m_channelVectorSize+m_layerParameters.m_outputSize);
}
