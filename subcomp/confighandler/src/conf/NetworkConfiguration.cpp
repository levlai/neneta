#include <NetworkConfiguration.h>
#include <InputLayer.h>
#include <ConvLayer.h>
#include <FCLayer.h>
#include <SoftMaxLayer.h>
#include <ProjectionLayer.h>
#include <ErrorCalculationLayer.h>
#include <SpectralPoolingLayer.h>


/*
 * TODO:
 * - find all layer siblings from the beginning
 * - format the memory based on the network configuration
 * - input layer - copies the data (minibatch) to the gpgpu memory
 *               -
 * - dft layer - perform
 *
 *
 *
 */

using namespace neneta;
using namespace neneta::conf;

NetworkConfiguration::NetworkConfiguration(const conf::ConfigurationReader& envReader)
    : Configuration(envReader.getStringParameter("configuration.neuralnetwork.networkparamsconfig"))
    , m_networkId(envReader.getStringParameter("configuration.neuralnetwork.networkparamsconfig"))
    , m_oclContext(envReader)
{    
    m_oclContext.printInfo();
    gpu::OpenCLProgram oclProgram(envReader);
    if(oclProgram.compile(m_oclContext))
    {        
        parseConfiguration(envReader, oclProgram);
        configureForwardPropagation();
        configureBackPropagation();

        BOOST_LOG_TRIVIAL(debug) << "Updating persistance layer list";
        std::vector<std::shared_ptr<net::IPersistedLayer>> persistedLayers;
        updatePersistedLayers(persistedLayers);

        updateParamSetsMap(std::piecewise_construct,
                       std::forward_as_tuple(m_networkId),
                       std::forward_as_tuple(std::dynamic_pointer_cast<gpu::IOpenCLInputExecutionPlan>(*std::begin(m_layers)),
                                             *std::prev(std::end(m_layers)),
                                             *std::next(std::begin(m_layers)),
                                             std::dynamic_pointer_cast<gpu::IOpenCLOutputExecutionPlan>(*std::prev(std::end(m_layers))),
                                             persistedLayers));
    }
}

NetworkConfiguration::~NetworkConfiguration()
{

}

void NetworkConfiguration::updatePersistedLayers(std::vector<std::shared_ptr<net::IPersistedLayer>>& vec)
{
    for(auto& layer : m_layers)
    {
        if(std::shared_ptr<net::IPersistedLayer> perLayer = std::dynamic_pointer_cast<net::IPersistedLayer>(layer))
        {
            BOOST_LOG_TRIVIAL(debug) << "Found persistence layer";
            vec.emplace_back(perLayer);
        }
    }
}

std::shared_ptr<gpu::OpenCLBasicExecutionPlan> NetworkConfiguration::getLayerBasedOnId(const std::string& layerId,
                                                                                       const ConfigurationReader& conf,
                                                                                       const conf::ConfigurationReader& envReader,
                                                                                       const gpu::OpenCLProgram& oclProgram)
{
    std::shared_ptr<gpu::OpenCLBasicExecutionPlan> newLayer;
    std::string layerType = conf.getSiblingConfiguration("neneta", "layer", "id", layerId).getStringAttribute("layer", "type");
    if(layerType.compare("input")==0)
    {
        BOOST_LOG_TRIVIAL(info) << "Adding InputLayer - " << layerId;
        newLayer.reset(new net::InputLayer(layerId, envReader, oclProgram, m_oclContext));
    }
    else if(layerType.compare("conv")==0)
    {
        BOOST_LOG_TRIVIAL(info) << "Adding ConvLayer - " << layerId;
        newLayer.reset(new net::ConvLayer(layerId, envReader, oclProgram, m_oclContext));
    }
    else if(layerType.compare("spectralpool")==0)
    {
        BOOST_LOG_TRIVIAL(info) << "Adding SpectralPoolingLayer - " << layerId;
        newLayer.reset(new net::SpectralPoolingLayer(layerId, envReader, oclProgram, m_oclContext));
    }
    else if(layerType.compare("fc")==0)
    {
        BOOST_LOG_TRIVIAL(info) << "Adding FCLayer - " << layerId;
        newLayer.reset(new net::FCLayer(layerId, envReader, oclProgram, m_oclContext));
    }
    else if(layerType.compare("projection")==0)
    {
        BOOST_LOG_TRIVIAL(info) << "Adding ProjectionLayer - " << layerId;
        newLayer.reset(new net::ProjectionLayer(layerId, envReader, oclProgram, m_oclContext));
    }
    else if(layerType.compare("softmax")==0)
    {
        BOOST_LOG_TRIVIAL(info) << "Adding SoftMaxLayer - " << layerId;
        newLayer.reset(new net::SoftMaxLayer(layerId, envReader, oclProgram, m_oclContext));
    }
    else if(layerType.compare("errorcalc")==0)
    {
        BOOST_LOG_TRIVIAL(info) << "Adding ErrorCalculationLayer - " << layerId;
        newLayer.reset(new net::ErrorCalculationLayer(layerId, envReader, oclProgram, m_oclContext));
    }
    else
    {
        BOOST_LOG_TRIVIAL(error) << "Unknown layer type";
    }
    return newLayer;
}

std::string NetworkConfiguration::findInputLayerId(const ConfigurationReader& conf)
{
    std::string inputLayerId;
    std::vector<ConfigurationReader> inputLayer = conf.getSiblingsConfiguration("neneta", "layer", "type", "input");
    if(inputLayer.size() == 1)
    {
        inputLayerId = inputLayer.front().getStringAttribute("layer", "id");
    }

    return inputLayerId;
}

std::string NetworkConfiguration::findNextLayer(const std::string& previousLayer, const std::vector<ConfigurationReader>& allLayers)
{
    std::string nextLayer;
    for(const auto& layerConf : allLayers)
    {
        std::string possibleNextLayer = layerConf.getStringParameter("layer.input");
        if(possibleNextLayer.compare(previousLayer)==0)
        {
            nextLayer = layerConf.getStringAttribute("layer", "id");;
            break;
        }
    }
    return nextLayer;
}

void NetworkConfiguration::parseConfiguration(const conf::ConfigurationReader& envReader, const gpu::OpenCLProgram& oclProgram)
{
    std::string layerId;
    ConfigurationReader layerConfiguration(*this);
    std::vector<ConfigurationReader> allLayers = layerConfiguration.getSiblingsConfiguration("neneta", "layer");
    if(!(layerId = findInputLayerId(layerConfiguration)).empty())
    {
        do
        {
            m_layers.emplace_back(std::static_pointer_cast<gpu::OpenCLExecutionPlan>(getLayerBasedOnId(layerId, layerConfiguration, envReader, oclProgram)));
        }
        while(!(layerId = findNextLayer(layerId,  allLayers)).empty());
    }
}

void NetworkConfiguration::configureForwardPropagation()
{
    BOOST_LOG_TRIVIAL(debug) << "Configuring forward propagation";
    auto start = std::begin(m_layers);
    auto end = std::end(m_layers);
    for(auto left = start, right = std::next(start); right != end; ++left, ++right)
    {
        **left >> **right;
    }
}

void NetworkConfiguration::configureBackPropagation()
{
    BOOST_LOG_TRIVIAL(debug) << "Configuring back propagation";
    auto start = std::prev(std::end(m_layers));
    auto end = std::begin(m_layers);
    for(auto left = std::prev(start), right = start; left != end; --left, --right)
    {
        **left << **right;
    }

}

