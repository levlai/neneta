#pragma once

#include <Configuration.h>
#include <LayerConfiguration.h>
#include <string>

namespace neneta
{
namespace conf
{

struct ErrorCalculationLayerParams : public LayerParams
{
    ErrorCalculationLayerParams(unsigned int channels, const std::string& errorFunc, const std::string& id)
        : LayerParams(ELayerType::ERROR_CALC)
        , m_channels(channels)
        , m_errorFunc(errorFunc)
        , m_id(id)
    {}
    ErrorCalculationLayerParams()
        : LayerParams(ELayerType::ERROR_CALC)
        , m_channels(0)
        , m_errorFunc("")
        , m_id("")
    {}

    unsigned int m_channels;
    std::string m_errorFunc;
    std::string m_id;
};


class ConfigurationReader;

class ErrorCalculationLayerConfiguration : public Configuration<ErrorCalculationLayerParams>
{
public:
    ErrorCalculationLayerConfiguration(const std::string& layerId, const ConfigurationReader& confReader);
};

}
}
