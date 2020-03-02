#pragma once

#include <memory>
#include <IImageDispatcher.h>

namespace neneta
{
namespace conf
{
class ConfigurationReader;
}

namespace imh
{

class IImageProcessor;

class ImageDispatcher : public IImageDispatcher//: public IPersistanceClient, public ThreadLoop
{
public:
    ImageDispatcher(const conf::ConfigurationReader& confReader);
    ~ImageDispatcher();

    //IPersistanceClient interfaces
//    void storeDatabase(const std::string& path) override;
//    void restoreDatabase(const std::string& path) override;
//    void storeState(const std::string& path) override;
//    void restoreState(const std::string& path) override;

    void init();
    void registerImageProcessor(std::shared_ptr<IImageProcessor> imgProcessor);
    void unregisterImageProcessor(std::shared_ptr<IImageProcessor> imgProcessor);
    void wait();

private:
    void dispatchImageTask(std::shared_ptr<IImageProcessor> imgProcessor, size_t threadId); // one per image processor
    void readImageTask();     // one
    void updateDbTask();      // one

private:
    class ImageProcessorImpl;
    std::shared_ptr<ImageProcessorImpl> m_impl;
};


void initdb(const conf::ConfigurationReader& confReader);

}
}
