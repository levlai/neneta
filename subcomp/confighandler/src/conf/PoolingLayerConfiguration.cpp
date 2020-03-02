#include <PoolingLayerConfiguration.h>
#include <ConfigurationReader.h>

using namespace neneta::conf;

PoolingLayerConfiguration::PoolingLayerConfiguration(const std::string& layerId, const ConfigurationReader& confReader)
    : Configuration(confReader.getStringParameter("configuration.neuralnetwork.networkparamsconfig"))
{
    ConfigurationReader networkParamsConfig(*this);
    ConfigurationReader layerConf = networkParamsConfig.getSiblingConfiguration("neneta", "layer", "type", "spectralpool", "id", layerId);
    updateParamSetsMap(std::piecewise_construct,
                   std::forward_as_tuple(layerId),
                   std::forward_as_tuple(//layerConf.getInt32Parameter("layer.input"),
                                         layerConf.getInt32Parameter("layer.inputsize"),
                                         layerConf.getInt32Parameter("layer.outputsize"),
                                         layerConf.getInt32Parameter("layer.channels"),
                                         layerId));
}
