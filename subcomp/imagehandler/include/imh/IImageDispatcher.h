#pragma once

#include <memory>

namespace neneta
{
namespace conf
{
class ConfigurationReader;
}

namespace imh
{

class IImageProcessor;

class IImageDispatcher
{
public:
    virtual void init() = 0;
    virtual void registerImageProcessor(std::shared_ptr<IImageProcessor> imgProcessor) = 0;
    virtual void unregisterImageProcessor(std::shared_ptr<IImageProcessor> imgProcessor) = 0;
    virtual void wait() = 0;

    virtual ~IImageDispatcher() {}
};

}
}
