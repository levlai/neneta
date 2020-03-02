#pragma once

#include <Types.h>

namespace neneta
{

namespace gpu
{

class IOpenCLOutputExecutionPlan
{
public:
    virtual cmn::GPUFLOAT getLoss() const = 0;
    virtual cmn::GPUFLOAT getAccuracy() const = 0;

protected:
    ~IOpenCLOutputExecutionPlan() {}
};


} // gpu
} // neneta
