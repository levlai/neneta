#pragma once

#include <cstdint>
#include <ConfigurationReader.h>
#include <Configuration.h>
#include <OpenCLExecutionPlan.h>
#include <OpenCLContext.h>
#include <IOpenCLInputExecutionPlan.h>
#include <IOpenCLOutputExecutionPlan.h>
#include <IPersistedLayer.h>
#include <OpenCLBasicExecutionPlan.h>

namespace neneta
{
namespace conf
{

struct NetworkExecutionPlan
{
    NetworkExecutionPlan(std::shared_ptr<gpu::IOpenCLInputExecutionPlan> in,
                         std::shared_ptr<gpu::OpenCLExecutionPlan> run,
                         std::shared_ptr<gpu::OpenCLExecutionPlan>  bcRun,
                         std::shared_ptr<gpu::IOpenCLOutputExecutionPlan> out,
                         std::vector<std::shared_ptr<net::IPersistedLayer>> perLayers)
        : m_input(in)
        , m_fwRun(run)
        , m_bcRun(bcRun)
        , m_output(out)
        , m_persistedLayers(perLayers)
    {}
    std::shared_ptr<gpu::IOpenCLInputExecutionPlan> m_input;
    std::shared_ptr<gpu::OpenCLExecutionPlan> m_fwRun;
    std::shared_ptr<gpu::OpenCLExecutionPlan> m_bcRun;
    std::shared_ptr<gpu::IOpenCLOutputExecutionPlan> m_output;
    std::vector<std::shared_ptr<net::IPersistedLayer>> m_persistedLayers;
};

class NetworkConfiguration : public Configuration<NetworkExecutionPlan>
{
public:
    NetworkConfiguration(const conf::ConfigurationReader& confReader);
    ~NetworkConfiguration();

    std::string getId() const {return m_networkId;}
    const gpu::OpenCLContext& getOpenCLContext() const { return m_oclContext; }

private:
    void updatePersistedLayers(std::vector<std::shared_ptr<net::IPersistedLayer>>& vec);
    void parseConfiguration(const conf::ConfigurationReader& envReader, const gpu::OpenCLProgram& oclProgram);
    void configureForwardPropagation();
    void configureBackPropagation();
    std::shared_ptr<gpu::OpenCLBasicExecutionPlan> getLayerBasedOnId(const std::string& layerId,
                                                                     const ConfigurationReader& conf,
                                                                     const conf::ConfigurationReader& envReader,
                                                                     const gpu::OpenCLProgram& oclProgram);
    std::string findInputLayerId(const ConfigurationReader& conf);
    std::string findNextLayer(const std::string& previousLayer, const std::vector<ConfigurationReader>& conf);
private:
    std::string m_networkId;
    std::vector<std::shared_ptr<gpu::OpenCLExecutionPlan>> m_layers;
    gpu::OpenCLContext m_oclContext;

};

}
}
