#include <gtest/gtest.h>
#include <IConfiguration.h>
#include <ConfigurationReader.h>
#include <Logging.h>
#include <boost/log/trivial.hpp>
#include <MNISTImageDispatcher.h>
#include <DispatcherEvent.h>
#include <IImageProcessor.h>
#include <NetworkConfiguration.h>
#include <DispatcherEvent.h>
#include <InputLayer.h>

using namespace neneta;
extern conf::ConfigurationReader envReader;

class Neneta : public imh::IImageProcessor
{
public:
    Neneta(const conf::ConfigurationReader& confReader);
    ~Neneta();

private:
    void processTrainingEvent(std::shared_ptr<imh::DispatcherEvent> event);
    void processValidationEvent(std::shared_ptr<imh::DispatcherEvent> event);
    void processTestEvent(std::shared_ptr<imh::DispatcherEvent> event);
    void updateGraph(const cmn::GPUFLOAT newErrorMeas);

private:
    const conf::ConfigurationReader& m_confReader;
    conf::NetworkConfiguration m_netConfig;
    conf::NetworkExecutionPlan m_executionPlan;
};


Neneta::Neneta(const conf::ConfigurationReader& confReader)
    : m_confReader(confReader)
    , m_netConfig(m_confReader)
    , m_executionPlan(m_netConfig.getParamSet(m_netConfig.getId()))
{
}

Neneta::~Neneta()
{
}

void Neneta::processValidationEvent(std::shared_ptr<imh::DispatcherEvent> event)
{

}

void Neneta::processTestEvent(std::shared_ptr<imh::DispatcherEvent> event)
{

}

void Neneta::processTrainingEvent(std::shared_ptr<imh::DispatcherEvent> event)
{
    static int i = 0;
    switch(event->getEventType())
    {
        case imh::DispatcherEvent::Type::IMAGE: //new image read from db, store it in input, and run nn
        {
            BOOST_LOG_TRIVIAL(info) << "NewImage " << ++i;
            std::unique_ptr<ip::Image<std::uint8_t, std::int16_t>> img = std::move(std::static_pointer_cast<imh::ImageEvent<std::uint8_t>>(event)->m_data);
            //ip::Image<cmn::GPUFLOAT, std::int16_t> complexImg = img->convertToComplex(256);
            ip::Image<cmn::GPUFLOAT, std::int16_t> complexImg = img->convertToRealFloat(256);
            std::static_pointer_cast<net::InputLayer>(m_executionPlan.m_input)->setInput(complexImg);
            m_executionPlan.m_fwRun->runFwdPropagation(m_netConfig.getOpenCLContext());
            m_executionPlan.m_fwRun->printFwdProfilingInfo();
            updateGraph(m_executionPlan.m_output->getLoss());
            break;
        }
        case imh::DispatcherEvent::Type::ENDOFDB:
            BOOST_LOG_TRIVIAL(info) << "Processing finished";
            break;
        case imh::DispatcherEvent::Type::STARTOFBATCH: //reset input
            BOOST_LOG_TRIVIAL(info) << "Start of new batch";
            //here is planned to store parameters
            //visualize output etc.
            break;
        case imh::DispatcherEvent::Type::ENDOFBATCH: //execute network
            BOOST_LOG_TRIVIAL(info) << "End of batch, running backpropagation";
            m_executionPlan.m_bcRun->runBckPropagation(m_netConfig.getOpenCLContext());
            m_executionPlan.m_bcRun->printBckProfilingInfo();
            exit(1);
            break;
        default:
            break;
    }
}

void Neneta::updateGraph(const cmn::GPUFLOAT newErrorMeas)
{
    BOOST_LOG_TRIVIAL(info) << "Error = " << newErrorMeas;
    /*int main(int argc,char **argv) {

      // Read command line argument.
      cimg_usage("Simple plotter of mathematical formulas");
      const char *const formula = cimg_option("-f","sin(x/8) % cos(2*x)","Formula to plot");
      const cmn::GPUFLOAT x0 = cimg_option("-x0",-5.0f,"Minimal X-value");
      const cmn::GPUFLOAT x1 = cimg_option("-x1",5.0f,"Maximal X-value");
      const int resolution = cimg_option("-r",1024,"Plot resolution");
      const unsigned int nresolution = resolution>1?resolution:1024;
      const unsigned int plot_type = cimg_option("-p",1,"Plot type");
      const unsigned int vertex_type = cimg_option("-v",1,"Vertex type");

      // Create plot data.
      CImg<double> values(4,nresolution,1,1,0);
      const unsigned int r = nresolution - 1;
      cimg_forY(values,X) values(0,X) = x0 + X*(x1 - x0)/r;
      cimg::eval(formula,values).move_to(values);

      // Display interactive plot window.
      values.display_graph(formula,plot_type,vertex_type,"X-axis",x0,x1,"Y-axis");

      // Quit.
      return 0;
    }*/
}


TEST(NeuralNetworkTests, forward_propagation_test)
{
    //Start logging
    logging::Logging logging(envReader);
    logging.init();

    BOOST_LOG_TRIVIAL(info) << "**** Starting neneta ****";

    //Create dispatcher
    imh::MNISTImageDispatcher imageDispatcher(imh::MNISTImageDispatcher::DispatchType::TRAINING, envReader); // event dispatcher: begin, startofbatch, image1, image2, ... imange N, endofbatch, .., end

    //Attach image processor
    std::shared_ptr<Neneta> imageProcessor = std::make_shared<Neneta>(envReader);
    imageDispatcher.registerImageProcessor(imageProcessor);

    //Init and wait until dispatching is done
    imageDispatcher.init();
    imageDispatcher.wait();

    BOOST_LOG_TRIVIAL(info) << "**** Closing neneta ****";
}
