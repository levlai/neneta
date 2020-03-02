#pragma once

#include <IImageProcessor.h>
#include <NetworkConfiguration.h>
#include <OpenCLProgram.h>
#include <vector>
#include <Plotter.h>

namespace neneta
{
namespace imh
{
class DispatcherEvent;
}
namespace conf
{
class ConfigurationReader;
}


class Neneta : public imh::IImageProcessor
{
public:
    Neneta(const conf::ConfigurationReader& confReader);
    ~Neneta();

private:
    void processTrainingEvent(std::shared_ptr<imh::DispatcherEvent> event);
    void processValidationEvent(std::shared_ptr<imh::DispatcherEvent> event);
    void processTestEvent(std::shared_ptr<imh::DispatcherEvent> event);

    void saveNetworkConfiguration();
private:    
    const conf::ConfigurationReader& m_confReader;
    plot::Plotter m_trainingPlot;
    plot::Plotter m_validationPlot;
    conf::NetworkConfiguration m_netConfig;
    conf::NetworkExecutionPlan m_executionPlan;
    float m_loss;
    float m_accuracy;
    unsigned int m_samples;
};


}
