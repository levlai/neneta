#pragma once

#include <Types.h>

namespace neneta
{
namespace ip
{
template<typename T, typename Label>
class Image;
}
namespace gpu
{

class IOpenCLInputExecutionPlan
{
public:    
    typedef ip::Image<cmn::GPUFLOAT, std::int16_t> ImageType;

    virtual void setInput(ImageType& image) = 0;

protected:
    ~IOpenCLInputExecutionPlan() {}
};


} // gpu
} // neneta
