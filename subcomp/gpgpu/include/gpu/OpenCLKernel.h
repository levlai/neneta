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

struct OpenCLKernelParameters
{
    OpenCLKernelParameters() {}
    OpenCLKernelParameters(unsigned int gws1, unsigned int lws1) : m_gws(gws1)
    {
        if(lws1 != 0)
        {
            m_lws = cl::NDRange(lws1);
        }
    }
    OpenCLKernelParameters(unsigned int gws1, unsigned int gws2, unsigned int lws1, unsigned int lws2) : m_gws(gws1, gws2)
    {
        if(lws1 != 0 && lws2 != 0)
        {
            m_lws = cl::NDRange(lws1, lws2);
        }
    }
    OpenCLKernelParameters(unsigned int gws1, unsigned int gws2, unsigned int gws3,
                           unsigned int lws1, unsigned int lws2, unsigned int lws3) : m_gws(gws1, gws2, gws3), m_lws(lws1, lws2, lws3)
    {
        if(lws1 != 0 && lws2 != 0 && lws3 != 0)
        {
            m_lws = cl::NDRange(lws1, lws2, lws3);
        }
    }
    cl::NDRange m_gws; //512, za fftshift 512,256
    cl::NDRange m_lws; //512, za fftshift 512,256
};

class OpenCLKernel
{
public:
    OpenCLKernel(const cl::Program& program, const std::string& name, const OpenCLKernelParameters& parameters);
    ~OpenCLKernel();

    cl::Kernel& getKernel();
    std::string getKernelName() const;
    cl::NDRange getGWS() const;
    cl::NDRange getLWS() const;
private:
    cl::Kernel  m_kernel;
    cl::NDRange m_gws;
    cl::NDRange m_lws;    
};

}
}
