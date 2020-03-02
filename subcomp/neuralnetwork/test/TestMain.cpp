#include <boost/log/trivial.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <IConfiguration.h>
#include <ConfigurationReader.h>
#include <Logging.h>


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

EnvConfiguration envConfig;
conf::ConfigurationReader envReader(envConfig);

int main(int argc, char* argv[])
{
    testing::InitGoogleMock(&argc, argv);

    logging::Logging logging(envReader);
    logging.init();

    int success = RUN_ALL_TESTS();

    return success;
}

