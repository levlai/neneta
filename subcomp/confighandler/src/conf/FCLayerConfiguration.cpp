#include <FCLayerConfiguration.h>
#include <ConfigurationReader.h>

using namespace neneta::conf;

FCLayerConfiguration::FCLayerConfiguration(const std::string& layerId, const ConfigurationReader& confReader)
    : Configuration(confReader.getStringParameter("configuration.neuralnetwork.networkparamsconfig"))
{
    ConfigurationReader networkParamsConfig(*this);

    ConfigurationReader layerConf = networkParamsConfig.getSiblingConfiguration("neneta", "layer", "type", "fc", "id", layerId);
    std::string weightsType = layerConf.getStringParameter("layer.weightstype");
    FCParams::WeightsType wt = FCParams::WeightsType::REAL;
    if(weightsType.compare("complex")==0)
    {
        wt = FCParams::WeightsType::COMPLEX;
    }
    updateParamSetsMap(std::piecewise_construct,
                   std::forward_as_tuple(layerId),
                   std::forward_as_tuple(layerConf.getInt32Parameter("layer.channels"),
                                         layerConf.getInt32Parameter("layer.inputdim"),
                                         layerConf.getInt32Parameter("layer.inputsize"),
                                         layerConf.getInt32Parameter("layer.outputsize"),
                                         layerConf.getDoubleParameter("layer.weightsdev"),
                                         layerConf.getDoubleParameter("layer.weightsmean"),
                                         wt,
                                         layerConf.getDoubleParameter("layer.biasre"),
                                         layerConf.getDoubleParameter("layer.biasim"),
                                         layerConf.getStringParameter("layer.actfunc"),
                                         layerId));

}
