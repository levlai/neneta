#pragma once

#include <Configuration.h>
#include <LayerConfiguration.h>
#include <string>

namespace neneta
{
namespace conf
{

enum LabelCoding
{
    REAL,
    COMPLEX
};

struct InputLayerParams : public LayerParams
{
    InputLayerParams(unsigned int rpipesize,
                     unsigned int ipipesize,
                     unsigned int dim,
                     unsigned int size,
                     unsigned int outSize,
                     unsigned int channels,
                     const std::string& id,
                     LabelCoding lcode)
        : LayerParams(ELayerType::INPUT)
        , m_rPipeSize(rpipesize)
        , m_iPipeSize(ipipesize)
        , m_inputDim(dim)
        , m_inputSize(size)
        , m_outputSize(outSize)
        , m_inputChannels(channels)
        , m_id(id)
        , m_labelCoding(lcode)
    {}
    InputLayerParams()
        : LayerParams(ELayerType::INPUT)
        , m_rPipeSize(0)
        , m_iPipeSize(0)
        , m_inputDim(0)
        , m_inputSize(0)
        , m_outputSize(0)
        , m_inputChannels(0)
        , m_id("")
        , m_labelCoding(LabelCoding::REAL)
    {}

    unsigned int m_rPipeSize;
    unsigned int m_iPipeSize;
    unsigned int m_inputDim;
    unsigned int m_inputSize;
    unsigned int m_outputSize;
    unsigned int m_inputChannels;
    std::string m_id;
    LabelCoding m_labelCoding;
};


class ConfigurationReader;

class InputLayerConfiguration : public Configuration<InputLayerParams>
{
public:
    InputLayerConfiguration(const std::string& layerId, const ConfigurationReader& confReader);
};

}
}
