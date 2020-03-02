#include <Neneta.h>
#include <DispatcherEvent.h>
#include <ConfigurationReader.h>
#include <boost/log/trivial.hpp>
#include <InputLayer.h>
#include <Types.h>

using namespace neneta;

Neneta::Neneta(const conf::ConfigurationReader& confReader)
    : m_confReader(confReader)    
    , m_trainingPlot(confReader)
    , m_validationPlot(confReader)
    , m_netConfig(m_confReader)
    , m_executionPlan(m_netConfig.getParamSet(m_netConfig.getId()))
    , m_loss(0)
    , m_accuracy(0)
    , m_samples(0)
{    
}

Neneta::~Neneta()
{
}

#include <chrono>

void Neneta::processTrainingEvent(std::shared_ptr<imh::DispatcherEvent> event)
{
    static int im = 0;
    switch(event->getEventType())
    {
        case imh::DispatcherEvent::Type::STARTOFDB:
            BOOST_LOG_TRIVIAL(info) << "Start of training epoch";
            break;
        case imh::DispatcherEvent::Type::IMAGE: //new image read from db, store it in input, and run nn
        {                        
          //  auto start = std::chrono::high_resolution_clock::now();
            std::unique_ptr<ip::Image<std::uint8_t, std::int16_t>> img = std::move(std::static_pointer_cast<imh::ImageEvent<std::uint8_t>>(event)->m_data);
            ip::Image<float, std::int16_t> complexImg = img->convertToComplex(255);
          //  ip::Image<cmn::GPUFLOAT, std::int16_t> complexImg = img->convertToRealFloat(255);
            std::static_pointer_cast<net::InputLayer>(m_executionPlan.m_input)->setInput(complexImg);
            m_executionPlan.m_fwRun->runFwdPropagation(m_netConfig.getOpenCLContext());
            m_executionPlan.m_fwRun->wait();
            m_loss += m_executionPlan.m_output->getLoss();
            m_accuracy += m_executionPlan.m_output->getAccuracy();
            m_samples++;            
            if((++im)%100 == 0)
            {
                m_trainingPlot.plotNewPoint(m_loss/m_samples);
            }
            //BOOST_LOG_TRIVIAL(info) << "Processed image -> " << m_samples << " loss = " << m_loss/m_samples;
         //   auto diff = std::chrono::high_resolution_clock::now() - start;
         //   BOOST_LOG_TRIVIAL(info) << "Fwd exec time: " << std::chrono::duration_cast<std::chrono::microseconds>(diff).count() << " us";
            break;
        }
        case imh::DispatcherEvent::Type::STARTOFBATCH: //reset input            
            break;
        case imh::DispatcherEvent::Type::ENDOFBATCH: //execute network
        {
          //  auto start = std::chrono::high_resolution_clock::now();
            m_executionPlan.m_bcRun->runBckPropagation(m_netConfig.getOpenCLContext());
            m_executionPlan.m_bcRun->wait();
         //   auto diff = std::chrono::high_resolution_clock::now() - start;
         //   BOOST_LOG_TRIVIAL(info) << "Bck exec time: " << std::chrono::duration_cast<std::chrono::microseconds>(diff).count() << " us";
            break;
        }
        case imh::DispatcherEvent::Type::ENDOFDB:
            BOOST_LOG_TRIVIAL(info) << "End of training epoch, loss = " << m_loss/m_samples << "\taccuracy = " << 100*m_accuracy/m_samples << "%";
            saveNetworkConfiguration();
            m_trainingPlot.startNewPlotLine();
            im = 0;
            m_loss = 0;
            m_accuracy = 0;
            m_samples = 0;
        default:
            break;
    }
}

void Neneta::processValidationEvent(std::shared_ptr<imh::DispatcherEvent> event)
{
    switch(event->getEventType())
    {
        case imh::DispatcherEvent::Type::STARTOFDB:
            BOOST_LOG_TRIVIAL(info) << "Start of validation";
            break;
        case imh::DispatcherEvent::Type::IMAGE:
        {
            std::unique_ptr<ip::Image<std::uint8_t, std::int16_t>> img = std::move(std::static_pointer_cast<imh::ImageEvent<std::uint8_t>>(event)->m_data);
            ip::Image<float, std::int16_t> complexImg = img->convertToComplex(255);
          //  ip::Image<cmn::GPUFLOAT, std::int16_t> complexImg = img->convertToRealFloat(255);
            std::static_pointer_cast<net::InputLayer>(m_executionPlan.m_input)->setInput(complexImg);
            m_executionPlan.m_fwRun->runFwdPropagation(m_netConfig.getOpenCLContext());
            m_executionPlan.m_fwRun->wait();
            m_loss += m_executionPlan.m_output->getLoss();
            m_accuracy += m_executionPlan.m_output->getAccuracy();
            m_samples++;
            break;
        }
        case imh::DispatcherEvent::Type::STARTOFBATCH:
            break;
        case imh::DispatcherEvent::Type::ENDOFBATCH:
            break;
        case imh::DispatcherEvent::Type::ENDOFDB:
            BOOST_LOG_TRIVIAL(info) << "End of validation, loss = " << m_loss/m_samples << "\taccuracy = " << 100*m_accuracy/m_samples << "%";
            m_validationPlot.plotNewPoint(m_loss/m_samples);
            m_loss = 0;
            m_accuracy = 0;
            m_samples = 0;
        default:
            break;
    }
}

void Neneta::processTestEvent(std::shared_ptr<imh::DispatcherEvent> event)
{
    static int im = 0;
    switch(event->getEventType())
    {
        case imh::DispatcherEvent::Type::STARTOFDB:
            BOOST_LOG_TRIVIAL(info) << "Start of test";
            break;
        case imh::DispatcherEvent::Type::IMAGE:
        {
            std::unique_ptr<ip::Image<std::uint8_t, std::int16_t>> img = std::move(std::static_pointer_cast<imh::ImageEvent<std::uint8_t>>(event)->m_data);
            ip::Image<float, std::int16_t> complexImg = img->convertToComplex(255);
        //    ip::Image<cmn::GPUFLOAT, std::int16_t> complexImg = img->convertToRealFloat(255);
            std::static_pointer_cast<net::InputLayer>(m_executionPlan.m_input)->setInput(complexImg);
            m_executionPlan.m_fwRun->runFwdPropagation(m_netConfig.getOpenCLContext());
            m_executionPlan.m_fwRun->wait();
            m_accuracy += m_executionPlan.m_output->getAccuracy();
            m_samples++;
            break;
        }
        case imh::DispatcherEvent::Type::STARTOFBATCH:
            break;
        case imh::DispatcherEvent::Type::ENDOFBATCH:
            break;
        case imh::DispatcherEvent::Type::ENDOFDB:
            BOOST_LOG_TRIVIAL(info) << "End of test, accuracy = " << 100*m_accuracy/m_samples << "%";
            m_loss = 0;
            m_accuracy = 0;
            m_samples = 0;
        default:
            break;
    }
}

void Neneta::saveNetworkConfiguration()
{
    for(auto& persistedLayer : m_executionPlan.m_persistedLayers)
    {
        persistedLayer->store();
    }
}
