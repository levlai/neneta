#include <OpenCLKernel.h>
#include <ConfigurationReader.h>
#include <boost/log/trivial.hpp>

using namespace neneta;
using namespace neneta::gpu;


OpenCLKernel::OpenCLKernel(const cl::Program& program, const std::string& name, const OpenCLKernelParameters& parameters)
    : m_kernel(program, name.c_str())
    , m_gws(parameters.m_gws)
    , m_lws(parameters.m_lws)
{
}

OpenCLKernel::~OpenCLKernel()
{

}

cl::Kernel& OpenCLKernel::getKernel()
{
    return m_kernel;
}

std::string OpenCLKernel::getKernelName() const
{
    return m_kernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
}

cl::NDRange OpenCLKernel::getGWS() const
{
    return m_gws;
}

cl::NDRange OpenCLKernel::getLWS() const
{
    return m_lws;
}
