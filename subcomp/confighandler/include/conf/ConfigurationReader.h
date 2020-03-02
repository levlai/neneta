#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace neneta
{
namespace conf
{

class IConfiguration;

class ConfigurationReader
{
public:
    ConfigurationReader();
    ConfigurationReader(const IConfiguration& configuration);
    ConfigurationReader(const std::string& configuration);
    ConfigurationReader& operator=(const std::string& configuration);

    double          getDoubleParameter(const std::string& parameterName) const;
    std::int32_t    getInt32Parameter(const std::string& parameterName) const;
    std::string     getStringParameter(const std::string& parameterName) const;
    std::string     getStringAttribute(const std::string& parameterName, const std::string& attributeName) const;
    bool            getBooleanParameter(const std::string& parameterName) const;
    std::vector<ConfigurationReader>    getSiblingsConfiguration(const std::string& siblingsPath, const std::string& siblingName) const;
    std::vector<ConfigurationReader>    getSiblingsConfiguration(const std::string& siblingsPath, const std::string& siblingName,
                                                                 const std::string& attributeName, const std::string& attributeValue) const;
    ConfigurationReader getSiblingConfiguration(const std::string& siblingsPath, const std::string& siblingName,
                                                const std::string& attribute1Name, const std::string& attribute1Value) const;
    ConfigurationReader getSiblingConfiguration(const std::string& siblingsPath, const std::string& siblingName,
                                                const std::string& attribute1Name, const std::string& attribute1Value,
                                                const std::string& attribute2Name, const std::string& attribute2Value) const;

private:
    class ConfigImpl;
    std::shared_ptr<ConfigImpl> m_configImpl;
};

}
}
