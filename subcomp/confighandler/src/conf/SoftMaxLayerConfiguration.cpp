#include <SoftMaxLayerConfiguration.h>
#include <ConfigurationReader.h>

using namespace neneta::conf;

SoftMaxLayerConfiguration::SoftMaxLayerConfiguration(const std::string& layerId, const ConfigurationReader& confReader)
    : Configuration(confReader.getStringParameter("configuration.neuralnetwork.networkparamsconfig"))
{
    ConfigurationReader networkParamsConfig(*this);

    ConfigurationReader layerConf = networkParamsConfig.getSiblingConfiguration("neneta", "layer", "type", "softmax", "id", layerId);
    updateParamSetsMap(std::piecewise_construct,
                   std::forward_as_tuple(layerId),
                   std::forward_as_tuple(layerConf.getInt32Parameter("layer.channels"),
                                         layerConf.getInt32Parameter("layer.inputdim"),
                                         layerConf.getInt32Parameter("layer.inputsize"),
                                         layerConf.getInt32Parameter("layer.outputsize"),
                                         layerConf.getStringParameter("layer.actfunc"),
                                         layerConf.getDoubleParameter("layer.weightsdev"),
                                         layerConf.getDoubleParameter("layer.weightsmean"),
                                         layerConf.getDoubleParameter("layer.bias"),
                                         layerId));

}
