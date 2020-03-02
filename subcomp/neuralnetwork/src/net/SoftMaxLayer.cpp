#include <SoftMaxLayer.h>
#include <OpenCLContext.h>
#include "Utils.h"

using namespace neneta;
using namespace neneta::net;



SoftMaxLayer::SoftMaxLayer(const std::string& layerId, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext)
    : gpu::OpenCLExecutionPlan(layerId, confReader, program)
    , m_persistance(confReader)
    , m_layerParameters(conf::SoftMaxLayerConfiguration(getId(), confReader).getParamSet(getId()))
    , m_channelVectorSize(std::pow(m_layerParameters.m_inputSize, m_layerParameters.m_inputDim))
    , m_numOfNeuronInputs(m_channelVectorSize*m_layerParameters.m_channels)    
    , m_clContext(oclContext)
    , m_clProgram(program)        
    , m_maxWGSize(m_clContext.getDevice().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>())
    , m_numOfWGPerNeuron(std::ceil(static_cast<cmn::GPUFLOAT>(m_numOfNeuronInputs)/static_cast<cmn::GPUFLOAT>(m_maxWGSize)))
    , m_biases(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT)),
               cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT)),
               m_layerParameters.m_outputSize)
    , m_deltas(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT2)),
               cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT2)),
               m_layerParameters.m_outputSize)
    , m_layerInput(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_numOfNeuronInputs*sizeof(cmn::GPUFLOAT)),
                   cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_numOfNeuronInputs*sizeof(cmn::GPUFLOAT)),
                   m_numOfNeuronInputs)
    , m_regularizationTerm(m_clContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT))
    , m_maxValue(m_clContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT))
    , m_reResult(m_clContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT)*m_layerParameters.m_outputSize)
    , m_imResult(m_clContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT)*m_layerParameters.m_outputSize)
    , m_reSmallTemp(m_clContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT)*m_numOfWGPerNeuron)
    , m_imSmallTemp(m_clContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT)*m_numOfWGPerNeuron)
{        
    restore();

    cmn::GPUFLOAT2 initValue;
    initValue.s[0] = 1;
    initValue.s[1] = 0;
    initBuffers<cmn::GPUFLOAT2>(*this, m_clContext.getCommandQueue(), m_deltas.m_size, m_deltas.m_re, initValue, m_deltas.m_im, initValue);
}

SoftMaxLayer::SoftMaxLayer(const conf::SoftMaxParams& layerParams, const std::vector<cmn::GPUFLOAT>& weights, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext)
    : gpu::OpenCLExecutionPlan("softmaxlayer", confReader, program)
    , m_persistance(confReader)
    , m_layerParameters(layerParams)
    , m_channelVectorSize(std::pow(m_layerParameters.m_inputSize, m_layerParameters.m_inputDim))
    , m_numOfNeuronInputs(m_channelVectorSize*m_layerParameters.m_channels)    
    , m_clContext(oclContext)
    , m_clProgram(program)
    , m_maxWGSize(m_clContext.getDevice().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>())
    , m_numOfWGPerNeuron(std::ceil(static_cast<cmn::GPUFLOAT>(m_numOfNeuronInputs)/m_maxWGSize))
    , m_biases(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT)),
               cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT)),
               m_layerParameters.m_outputSize)
    , m_deltas(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT2)),
               cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT2)),
               m_layerParameters.m_outputSize)
    , m_layerInput(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_numOfNeuronInputs*sizeof(cmn::GPUFLOAT)),
                   cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_numOfNeuronInputs*sizeof(cmn::GPUFLOAT)),
                   m_numOfNeuronInputs)
    , m_regularizationTerm(m_clContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT))
    , m_maxValue(m_clContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT))
    , m_reResult(m_clContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT)*m_layerParameters.m_outputSize)
    , m_imResult(m_clContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT)*m_layerParameters.m_outputSize)
    , m_reSmallTemp(m_clContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT)*m_numOfWGPerNeuron)
    , m_imSmallTemp(m_clContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT)*m_numOfWGPerNeuron)
{
    std::vector<cmn::GPUFLOAT> ungroupedweights = weights;
    std::vector<cmn::GPUFLOAT> bias(m_layerParameters.m_outputSize, m_layerParameters.m_initBias);
    std::vector<cmn::GPUFLOAT> imbias(m_layerParameters.m_outputSize, 0);
    ungroupedweights.insert(ungroupedweights.end(), bias.begin(), bias.end());
    ungroupedweights.insert(ungroupedweights.end(), imbias.begin(), imbias.end());
    init(ungroupedweights);

    cmn::GPUFLOAT2 initValue;
    initValue.s[0] = 1;
    initValue.s[1] = 0;
    initBuffers<cmn::GPUFLOAT2>(*this, m_clContext.getCommandQueue(), m_deltas.m_size, m_deltas.m_re, initValue, m_deltas.m_im, initValue);
}

SoftMaxLayer::~SoftMaxLayer()
{
}

void SoftMaxLayer::setInput(gpu::BufferIO input)
{
    prepareBuffers(input);

    //here is m_layerInput reused for tmp storage
    BOOST_LOG_TRIVIAL(debug) << "SoftMaxLayer id - " << m_layerParameters.m_id << " setting kernels";
    for(unsigned int outChannel = 0; outChannel < m_layerParameters.m_outputSize; ++outChannel)
    { //for each neuron

        unsigned int inputSize = m_numOfNeuronInputs;
        BOOST_LOG_TRIVIAL(debug) << "SoftMaxLayer for neuron " << outChannel << " inputSize = " << inputSize << " calculating product";
        gpu::OpenCLKernelParameters vhpKernel(inputSize, 0);
        planFwd("vec_hadamard_product", vhpKernel, m_layerInput.m_re, m_layerInput.m_im,
                                      m_io.m_reShMem, m_io.m_imShMem,
                                      m_weights[outChannel].m_re, m_weights[outChannel].m_im);

        do
        {
            gpu::OpenCLKernelParameters kparams = calculateKernelParameters(inputSize, m_maxWGSize);
            planFwd("sum", kparams, static_cast<cl_int>(inputSize), m_reSmallTemp, m_imSmallTemp,
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
                assert(inputSize == 1 && "inputSize in SoftMaxLayer wrong!");                
                planFwd("addBias", gpu::OpenCLKernelParameters(1,1),
                        m_reResult, m_imResult,
                        m_reSmallTemp, m_imSmallTemp,
                        m_biases.m_re, m_biases.m_im,
                        static_cast<cl_int>(outChannel));
            }
        } while(inputSize > 1);
    }    

    //store activation potential to shmembkp
    planCopyBufferFwd(m_reResult, m_io.m_reShMemBkp, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT));
    planCopyBufferFwd(m_imResult, m_io.m_imShMemBkp, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT));

    //find max activation potential
    maxValue(*this, m_reResult, m_maxValue, m_reSmallTemp, m_layerParameters.m_outputSize, m_maxWGSize);

    //take an exp(res-max)
    planFwd("exponent", gpu::OpenCLKernelParameters(m_layerParameters.m_outputSize,0), m_reResult, m_maxValue);

    //calculate regularization term, exponents are in m_reResult
    unsigned int inputSize = m_layerParameters.m_outputSize;
    do
    {
        gpu::OpenCLKernelParameters kparams = calculateKernelParameters(inputSize, m_maxWGSize);
        planFwd("sumOfExponents", kparams, inputSize, m_imResult, m_reResult, cl::Local(sizeof(cmn::GPUFLOAT)*m_maxWGSize));

        inputSize = inputSize/m_maxWGSize + (inputSize%m_maxWGSize != 0);

        if(inputSize > 1)
        {
            planCopyBufferFwd(m_imResult, m_reResult, inputSize*sizeof(cmn::GPUFLOAT));
        }
        else
        {
            assert(inputSize == 1 && "inputSize in SoftMaxLayer wrong!");
            planCopyBufferFwd(m_imResult, m_regularizationTerm, sizeof(cmn::GPUFLOAT));
        }
    } while(inputSize > 1);

    //store input
    planCopyBufferFwd(m_io.m_reShMem, m_layerInput.m_re, m_numOfNeuronInputs*sizeof(cmn::GPUFLOAT));
    planCopyBufferFwd(m_io.m_imShMem, m_layerInput.m_im, m_numOfNeuronInputs*sizeof(cmn::GPUFLOAT));

    //run through activation function
    gpu::OpenCLKernelParameters kernel(m_layerParameters.m_outputSize, 0);
    planFwd(m_layerParameters.m_activationFunction.c_str(), kernel,
            m_io.m_reShMemBkp, m_io.m_imShMemBkp, //act pot
            m_io.m_reShMem, m_io.m_imShMem,       //output
            m_regularizationTerm, m_maxValue);

}

gpu::BufferIO SoftMaxLayer::getOutput()
{
    return m_io;
}

void SoftMaxLayer::calculateDeltas()
{
    planCopyBufferBck(m_io.m_reShMem, m_deltas.m_re, m_deltas.m_size*sizeof(cmn::GPUFLOAT));
    planCopyBufferBck(m_io.m_imShMem, m_deltas.m_im, m_deltas.m_size*sizeof(cmn::GPUFLOAT));
}

void SoftMaxLayer::calculateErrors()
{
    //Calculate errors to be propagated to the left layer
    calculateErrorsFC(*this, m_io, m_weights, m_deltas, m_layerParameters.m_outputSize, m_numOfNeuronInputs);
}

void SoftMaxLayer::updateWeights()
{
    updateWeightsFC(*this, m_weights, m_deltas, m_biases, m_layerInput, m_layerParameters.m_outputSize);
}

void SoftMaxLayer::readWeightsAndBiases(std::vector<cmn::GPUFLOAT>& weightsandbiases)
{
    unsigned int offset = 0;
    for(unsigned int i = 0; i < m_layerParameters.m_outputSize; ++i)
    {
        readFromBuffer(m_clContext.getCommandQueue(), m_weights[i].m_re, sizeof(cmn::GPUFLOAT)*m_numOfNeuronInputs, weightsandbiases.data() + offset);
        offset += m_numOfNeuronInputs;
    }
    //read biases
    readFromBuffer(m_clContext.getCommandQueue(), m_biases.m_re, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT), weightsandbiases.data() + offset);
}

void SoftMaxLayer::setBkpInput(gpu::BufferIO input)
{
    prepareBuffers(input);
    calculateDeltas();
    calculateErrors();
    updateWeights();
}

gpu::BufferIO SoftMaxLayer::getBkpOutput()
{
    return m_io;
}

void SoftMaxLayer::prepareBuffers(gpu::BufferIO input)
{
    m_io = input;
}

void SoftMaxLayer::store()
{
    std::vector<cmn::GPUFLOAT> weightsandbiases((m_numOfNeuronInputs+1)*m_layerParameters.m_outputSize);
    readWeightsAndBiases(weightsandbiases);
    m_persistance.storeFloatBlob(getId(), weightsandbiases);
}

void SoftMaxLayer::restore()
{
    std::vector<cmn::GPUFLOAT> ungroupedweights;
    m_persistance.restoreFloatBlob(getId(), ungroupedweights);
    if(ungroupedweights.empty())
    {
        BOOST_LOG_TRIVIAL(debug) << "SoftMaxLayer generating weights vector";
        ungroupedweights = generateRandomVector(m_numOfNeuronInputs*m_layerParameters.m_outputSize, m_layerParameters.m_weightsMean, m_layerParameters.m_initWeightsDeviation);
        std::vector<cmn::GPUFLOAT> bias(m_layerParameters.m_outputSize, m_layerParameters.m_initBias);
        ungroupedweights.insert(ungroupedweights.end(), bias.begin(), bias.end());
    }
    assert((ungroupedweights.size() == (m_numOfNeuronInputs+1)*m_layerParameters.m_outputSize) && "SoftMaxLayer::restore() doesn't work!");

    init(ungroupedweights);
}


void SoftMaxLayer::init(std::vector<cmn::GPUFLOAT>& rawdata)
{
    std::vector<cmn::GPUFLOAT> imags(m_numOfNeuronInputs, 0.0);
    std::vector<cmn::GPUFLOAT> imagsBias(m_layerParameters.m_outputSize, 0.0);
    for(unsigned int i = 0; i < m_layerParameters.m_outputSize; ++i)
    {
        m_weights.emplace_back(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , m_numOfNeuronInputs*sizeof(cmn::GPUFLOAT), (void*)(rawdata.data() + i*m_numOfNeuronInputs)),
                               cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , m_numOfNeuronInputs*sizeof(cmn::GPUFLOAT), imags.data()),
                               m_numOfNeuronInputs);
    }

    //init biases
    writeToBuffer(m_clContext.getCommandQueue(), m_biases.m_re, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT), rawdata.data() + m_numOfNeuronInputs*m_layerParameters.m_outputSize);
    writeToBuffer(m_clContext.getCommandQueue(), m_biases.m_im, m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT), imagsBias.data());
}

