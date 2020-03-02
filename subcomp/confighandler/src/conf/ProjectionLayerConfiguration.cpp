#include <ProjectionLayerConfiguration.h>
#include <ConfigurationReader.h>

using namespace neneta::conf;

ProjectionLayerConfiguration::ProjectionLayerConfiguration(const std::string& layerId, const ConfigurationReader& confReader)
    : Configuration(confReader.getStringParameter("configuration.neuralnetwork.networkparamsconfig"))
{
    ConfigurationReader networkParamsConfig(*this);
    ConfigurationReader layerConf = networkParamsConfig.getSiblingConfiguration("neneta", "layer", "type", "projection", "id", layerId);
    updateParamSetsMap(std::piecewise_construct,
                   std::forward_as_tuple(layerId),
                   std::forward_as_tuple(layerConf.getInt32Parameter("layer.channels"),
                                         layerConf.getStringParameter("layer.projectionfunc"),
                                         layerId));
}
