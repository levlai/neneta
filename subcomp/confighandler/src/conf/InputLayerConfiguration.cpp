#include <InputLayerConfiguration.h>
#include <ConfigurationReader.h>

using namespace neneta::conf;

InputLayerConfiguration::InputLayerConfiguration(const std::string& layerId, const ConfigurationReader& confReader)
    : Configuration(confReader.getStringParameter("configuration.neuralnetwork.networkparamsconfig"))
{
    ConfigurationReader networkParamsConfig(*this);
    ConfigurationReader layerConf = networkParamsConfig.getSiblingConfiguration("neneta", "layer", "type", "input", "id", layerId);
    updateParamSetsMap(std::piecewise_construct,
                   std::forward_as_tuple(layerId),
                   std::forward_as_tuple(layerConf.getInt32Parameter("layer.rpipesize"),
                                         layerConf.getInt32Parameter("layer.ipipesize"),
                                         layerConf.getInt32Parameter("layer.inputdim"),
                                         layerConf.getInt32Parameter("layer.inputsize"),
                                         layerConf.getInt32Parameter("layer.outputsize"),
                                         layerConf.getInt32Parameter("layer.inputchannels"),
                                         layerId,
                                         ((layerConf.getStringParameter("layer.labelcoding").compare("complex") == 0)? LabelCoding::COMPLEX : LabelCoding::REAL)));
}
