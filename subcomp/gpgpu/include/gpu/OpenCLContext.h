#pragma once

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

namespace neneta
{
namespace conf
{
class ConfigurationReader;
}

namespace gpu
{

class OpenCLContext
{
public:
    OpenCLContext(const conf::ConfigurationReader& confReader);
    ~OpenCLContext();

    const cl::Device& getDevice() const;
    const cl::Context& getContext() const;
    const cl::CommandQueue& getCommandQueue() const;

    void printInfo() const;

private:
    const conf::ConfigurationReader& m_confReader;
    cl::Device m_clDevice;
    cl::Context m_clContext;
    cl::CommandQueue m_clCommandQueue;

};

}
}
