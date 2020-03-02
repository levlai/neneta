#include <FourierConfiguration.h>
#include <ConfigurationReader.h>

using namespace neneta::conf;

FourierConfiguration::FourierConfiguration(const std::string& layerId, const ConfigurationReader& confReader)
    : Configuration(confReader.getStringParameter("configuration.neuralnetwork.networkparamsconfig"))
{
    ConfigurationReader networkParamsConfig(*this);
    ConfigurationReader layerConf = networkParamsConfig.getSiblingConfiguration("neneta", "layer", "type", "fft", "id", layerId);
    updateParamSetsMap(std::piecewise_construct,
                   std::forward_as_tuple(layerId),
                   std::forward_as_tuple(layerConf.getStringParameter("layer.input"),
                                         layerConf.getInt32Parameter("layer.size"),
                                         layerConf.getInt32Parameter("layer.channels"),
                                         layerId));
}
