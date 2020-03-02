#pragma once

#include <DispatcherEvent.h>
#include <memory>

namespace neneta
{
namespace imh
{

class IImageProcessor
{
public:    
    virtual void processTrainingEvent(std::shared_ptr<DispatcherEvent> event) = 0;
    virtual void processValidationEvent(std::shared_ptr<DispatcherEvent> event) = 0;
    virtual void processTestEvent(std::shared_ptr<DispatcherEvent> event) = 0;
protected:
    ~IImageProcessor() {}
};
}
}
