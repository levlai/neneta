#include <InputLayer.h>
#include <Image.h>
#include <OpenCLContext.h>
#include "Utils.h"

using namespace neneta;
using namespace neneta::net;


neneta::net::InputLayer::InputLayer(const std::string& layerId, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext)
    : gpu::OpenCLExecutionPlan(layerId, confReader, program)
    , m_clContext(oclContext)
    , m_layerParameters(conf::InputLayerConfiguration(getId(), confReader).getParamSet(getId()))
    , m_sharedGPUMemory()
{
    m_sharedGPUMemory.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_rPipeSize);
    m_sharedGPUMemory.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_iPipeSize);
    m_sharedGPUMemory.m_reShMemBkp = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_rPipeSize);
    m_sharedGPUMemory.m_imShMemBkp = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_iPipeSize);
    m_sharedGPUMemory.m_reDesired = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT)*m_layerParameters.m_outputSize);
    m_sharedGPUMemory.m_imDesired = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT)*m_layerParameters.m_outputSize);
}

InputLayer::InputLayer(const conf::InputLayerParams& params, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram &program, const gpu::OpenCLContext &oclContext)
    : gpu::OpenCLExecutionPlan("inputlayer", confReader, program)
    , m_clContext(oclContext)
    , m_layerParameters(params)
    , m_sharedGPUMemory()
{
    m_sharedGPUMemory.m_reShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_rPipeSize);
    m_sharedGPUMemory.m_imShMem = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_iPipeSize);
    m_sharedGPUMemory.m_reShMemBkp = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_rPipeSize);
    m_sharedGPUMemory.m_imShMemBkp = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_iPipeSize);
    m_sharedGPUMemory.m_reDesired = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT)*m_layerParameters.m_outputSize);
    m_sharedGPUMemory.m_imDesired = cl::Buffer(oclContext.getContext(), CL_MEM_READ_WRITE, sizeof(cmn::GPUFLOAT)*m_layerParameters.m_outputSize);
}

neneta::net::InputLayer::~InputLayer()
{
}

void InputLayer::setInput(ImageType& image)
{    
    int lblRealTrue = 1;
    int lblImagTrue = 0;
    int lblRealFalse = 0;
    int lblImagFalse = 0;
    if(m_layerParameters.m_labelCoding == conf::LabelCoding::COMPLEX)
    {
        lblRealTrue = 1;
        lblImagTrue = 1;
        lblRealFalse = -1;
        lblImagFalse = -1;
    }
    unsigned int imsizefloats = std::pow(m_layerParameters.m_inputSize, m_layerParameters.m_inputDim);
    std::vector<cmn::GPUFLOAT> reDes(m_layerParameters.m_outputSize, lblRealFalse);
    reDes[image.getLabel()] = lblRealTrue;
    std::vector<cmn::GPUFLOAT> imDes(m_layerParameters.m_outputSize, lblImagFalse);
    imDes[image.getLabel()] = lblImagTrue;

    writeToBuffer(m_clContext.getCommandQueue(), m_sharedGPUMemory.m_reShMem, imsizefloats *sizeof(cmn::GPUFLOAT), image.data(0));
    writeToBuffer(m_clContext.getCommandQueue(), m_sharedGPUMemory.m_imShMem, imsizefloats *sizeof(cmn::GPUFLOAT), image.data(1));
    writeToBuffer(m_clContext.getCommandQueue(), m_sharedGPUMemory.m_reDesired, reDes.size()*sizeof(cmn::GPUFLOAT), reDes.data());
    writeToBuffer(m_clContext.getCommandQueue(), m_sharedGPUMemory.m_imDesired, imDes.size()*sizeof(cmn::GPUFLOAT), imDes.data());

  //  gpuPrintMatrix(*this, m_clContext.getCommandQueue(), m_sharedGPUMemory.m_reShMem, imsizefloats, 1);
}


void neneta::net::InputLayer::setInput(gpu::BufferIO)
{    
}


gpu::BufferIO neneta::net::InputLayer::getOutput()
{
    return m_sharedGPUMemory;
}

void InputLayer::setBkpInput(gpu::BufferIO input)
{

}

gpu::BufferIO InputLayer::getBkpOutput()
{
    return m_sharedGPUMemory;
}

