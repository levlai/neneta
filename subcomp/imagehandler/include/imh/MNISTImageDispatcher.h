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

class MNISTImageDispatcher : public IImageDispatcher
{
public:
    enum DispatchType
    {
        TRAINING,
        VALIDATION,
        TEST
    };

    MNISTImageDispatcher(DispatchType type, const conf::ConfigurationReader& confReader);
    ~MNISTImageDispatcher();

    void init();
    void registerImageProcessor(std::shared_ptr<IImageProcessor> imgProcessor);
    void unregisterImageProcessor(std::shared_ptr<IImageProcessor> imgProcessor);
    void wait();

private:
    void dispatchImageTask(std::shared_ptr<IImageProcessor> imgProcessor, size_t threadId); // one per image processor
    void readImageTask();     // one
    std::streampos readTrainingImagesFile(std::ifstream& imagesfile);
    std::streampos readTrainingLabelsFile(std::ifstream& labelsfile);
    void readImages(std::ifstream& imagesfile, std::ifstream& labelsfile);

private:
    class ImageProcessorImpl;
    std::shared_ptr<ImageProcessorImpl> m_impl;
};


}
}
