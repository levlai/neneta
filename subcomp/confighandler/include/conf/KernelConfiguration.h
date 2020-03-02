#pragma once

#include <map>
#include <string>
#include <Configuration.h>

namespace neneta
{
namespace conf
{

struct KernelParams
{
    KernelParams(bool prof)
        : m_profilingEnabled(prof)
    {}
    bool m_profilingEnabled;
};

class ConfigurationReader;

class KernelConfiguration : public Configuration<KernelParams>
{
public:
    KernelConfiguration(const ConfigurationReader& confReader);

    unsigned int getLWS(unsigned int dim) const;
private:
    unsigned int m_lws1;
    unsigned int m_lws2;
    unsigned int m_lws3;
};

}
}
