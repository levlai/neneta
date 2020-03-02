#pragma once

#include <Configuration.h>
#include <LayerConfiguration.h>
#include <string>

namespace neneta
{
namespace conf
{

struct FourierParams : public LayerParams
{
    FourierParams(const std::string& input, unsigned int origsize, unsigned int channels, const std::string& id)
        : LayerParams(ELayerType::FOURIER)
        , m_input(input)
        , m_size(origsize)
        , m_channels(channels)
        , m_id(id)
    {}
    FourierParams()
        : LayerParams(ELayerType::FOURIER)
        , m_input("")
        , m_size(0)
        , m_channels(0)
        , m_id("")
    {}
    std::string  m_input;
    unsigned int m_size;
    unsigned int m_channels;
    std::string  m_id;
};


class ConfigurationReader;

class FourierConfiguration : public Configuration<FourierParams>
{
public:
    FourierConfiguration(const std::string& layerId, const ConfigurationReader& confReader);
};

}
}
