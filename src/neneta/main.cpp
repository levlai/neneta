#include <IConfiguration.h>
#include <ConfigurationReader.h>
#include <Logging.h>
#include <boost/log/trivial.hpp>
#include <MNISTImageDispatcher.h>
#include <DispatcherEvent.h>
#include <IImageProcessor.h>
#include "Neneta.h"

using namespace neneta;

struct EnvConfiguration : public conf::IConfiguration
{
    std::string getConfigurationFilename() const override
    {
#ifdef _WIN32
        return "configuration_win.xml";
#else
        return "configuration.xml";
#endif
    }
};

int main(int argc, char** argv)
{
    //Read configuration
    EnvConfiguration envConfig;
    conf::ConfigurationReader envReader(envConfig);

    //Start logging
    logging::Logging logging(envReader);
    logging.init();

    BOOST_LOG_TRIVIAL(info) << "**** Starting neneta ****";

    //Create neural network
    std::shared_ptr<Neneta> imageProcessor = std::make_shared<Neneta>(envReader);

    bool trainning = true;
    int training_epochs = 0;
    while(trainning)
    {
        imh::MNISTImageDispatcher imageDispatcherTraining(imh::MNISTImageDispatcher::TRAINING, envReader);
        imageDispatcherTraining.registerImageProcessor(imageProcessor);
        imageDispatcherTraining.init();
        imageDispatcherTraining.wait();

        imh::MNISTImageDispatcher imageDispatcherValidation(imh::MNISTImageDispatcher::VALIDATION, envReader);
        imageDispatcherValidation.registerImageProcessor(imageProcessor);
        imageDispatcherValidation.init();
        imageDispatcherValidation.wait();

        //Think of some stopping criteria

        if(++training_epochs == 10) trainning = false;
    }

    imh::MNISTImageDispatcher imageDispatcherTesting(imh::MNISTImageDispatcher::TEST, envReader);
    imageDispatcherTesting.registerImageProcessor(imageProcessor);    
    imageDispatcherTesting.init();
    imageDispatcherTesting.wait();

    BOOST_LOG_TRIVIAL(info) << "**** Closing neneta ****";

    return 0;
}
