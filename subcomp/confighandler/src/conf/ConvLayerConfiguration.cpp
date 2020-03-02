#include <ConvLayerConfiguration.h>
#include <ConfigurationReader.h>

using namespace neneta::conf;

ConvLayerConfiguration::ConvLayerConfiguration(const std::string& layerId, const ConfigurationReader& confReader)
    : Configuration(confReader.getStringParameter("configuration.neuralnetwork.networkparamsconfig"))
{
    ConfigurationReader networkParamsConfig(*this);
    //nenenta.layer.#id
    ConfigurationReader layerConf = networkParamsConfig.getSiblingConfiguration("neneta", "layer", "type", "conv", "id", layerId);
    std::string weightsType = layerConf.getStringParameter("layer.weightstype");
    ConvLayerParams::WeightsType wt = ConvLayerParams::WeightsType::REAL;
    if(weightsType.compare("complex")==0)
    {
        wt = ConvLayerParams::WeightsType::COMPLEX;
    }
    updateParamSetsMap(std::piecewise_construct,
                   std::forward_as_tuple(layerId),
                   std::forward_as_tuple(//layerConf.getStringParameter("layer.input"),
                                         layerConf.getInt32Parameter("layer.channels"),
                                         layerConf.getInt32Parameter("layer.kernels"),
                                         layerConf.getInt32Parameter("layer.kernelsize"),
                                         layerConf.getInt32Parameter("layer.stride"),
                                         layerConf.getInt32Parameter("layer.inputdim"),
                                         layerConf.getInt32Parameter("layer.inputsize"),                                         
                                         layerConf.getDoubleParameter("layer.weightsdev"),
                                         layerConf.getDoubleParameter("layer.weightsmean"),
                                         wt,
                                         layerConf.getDoubleParameter("layer.biasre"),
                                         layerConf.getDoubleParameter("layer.biasim"),
                                         layerConf.getStringParameter("layer.actfunc"),
                                         layerId));

}
