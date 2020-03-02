#pragma once

namespace neneta
{
namespace conf
{
class ConfigurationReader;
}

namespace logging
{

class Logging
{
public:
    Logging(const conf::ConfigurationReader& confReader);
    void init();
private:
    const conf::ConfigurationReader& m_confReader;
};

}
}
