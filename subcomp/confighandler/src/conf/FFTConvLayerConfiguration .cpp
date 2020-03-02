#include <FFTConvLayerConfiguration.h>
#include <ConfigurationReader.h>

using namespace neneta::conf;

FFTConvLayerConfiguration::FFTConvLayerConfiguration(const std::string& layerId, const ConfigurationReader& confReader)
    : Configuration(confReader.getStringParameter("configuration.neuralnetwork.networkparamsconfig"))
{
    ConfigurationReader networkParamsConfig(*this);
    //nenenta.layer.#id
    ConfigurationReader layerConf = networkParamsConfig.getSiblingConfiguration("neneta", "layer", "type", "spectralconv", "id", layerId);
    updateParamSetsMap(std::piecewise_construct,
                   std::forward_as_tuple(layerId),
                   std::forward_as_tuple(//layerConf.getStringParameter("layer.input"),
                                         layerConf.getInt32Parameter("layer.channels"),
                                         layerConf.getInt32Parameter("layer.kernels"),
                                         layerConf.getInt32Parameter("layer.kernelsize"),
                                         layerConf.getInt32Parameter("layer.inputsize"),
                                         //layerConf.getInt32Parameter("layer.actfunc"),
                                         layerConf.getDoubleParameter("layer.weightsdev"),
                                         layerConf.getDoubleParameter("layer.bias"),
                                         layerId));

}
