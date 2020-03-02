#pragma once

#include <Configuration.h>
#include <LayerConfiguration.h>
#include <string>

namespace neneta
{
namespace conf
{

struct ProjectionLayerParams : public LayerParams
{
    ProjectionLayerParams(unsigned int channels, const std::string& errorFunc, const std::string& id)
        : LayerParams(ELayerType::PROJECTION)
        , m_channels(channels)
        , m_projectionFunc(errorFunc)
        , m_id(id)
    {}
    ProjectionLayerParams()
        : LayerParams(ELayerType::PROJECTION)
        , m_channels(0)
        , m_projectionFunc("")
        , m_id("")
    {}

    unsigned int m_channels;
    std::string m_projectionFunc;
    std::string m_id;
};


class ConfigurationReader;

class ProjectionLayerConfiguration : public Configuration<ProjectionLayerParams>
{
public:
    ProjectionLayerConfiguration(const std::string& layerId, const ConfigurationReader& confReader);
};

}
}
