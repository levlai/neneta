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

class OpenCLContext;

class OpenCLProgram
{
public:
    OpenCLProgram(const conf::ConfigurationReader& confReader);
    ~OpenCLProgram();

    void addSource(const std::string& programSourceFilename);
    void addSources();
    bool compile(const OpenCLContext& openCLContext);
    const cl::Program& getProgram() const;
private:
    const conf::ConfigurationReader& m_confReader;
    std::vector<std::string> m_clProgramSources;
    cl::Program m_clProgram;
};

}
}
