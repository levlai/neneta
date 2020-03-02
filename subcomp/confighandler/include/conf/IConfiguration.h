#pragma once

#include <string>

namespace neneta
{
namespace conf
{

class IConfiguration
{
public:
    virtual std::string getConfigurationFilename() const = 0;

protected:
    ~IConfiguration() {}
};

}
}
