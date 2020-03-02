#pragma once

#include <Configuration.h>
#include <LayerConfiguration.h>
#include <string>

namespace neneta
{
namespace conf
{

struct SpectralPoolingParams : public LayerParams
{
    SpectralPoolingParams(unsigned int inputSize, unsigned int outputSize, unsigned int channels, const std::string& id)
        : LayerParams(ELayerType::SPECTRAL_POOL)
        , m_inputSize(inputSize)
        , m_outputSize(outputSize)
        , m_channels(channels)
        , m_id(id)
    {}
    unsigned int m_inputSize;
    unsigned int m_outputSize;
    unsigned int m_channels;
    std::string m_id;
};


class ConfigurationReader;

class PoolingLayerConfiguration : public Configuration<SpectralPoolingParams>
{
public:
    PoolingLayerConfiguration(const std::string& layerId, const ConfigurationReader& confReader);
};

}
}
