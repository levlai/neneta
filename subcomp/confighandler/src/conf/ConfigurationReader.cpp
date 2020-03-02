#include <ConfigurationReader.h>
#include <IConfiguration.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/log/trivial.hpp>
#include <fstream>
#include <iostream>
#include <vector>

namespace pt = boost::property_tree;
using namespace neneta::conf;

struct ConfigurationReader::ConfigImpl
{
    pt::ptree m_configCache;
};

ConfigurationReader::ConfigurationReader()
    : m_configImpl(std::make_shared<ConfigurationReader::ConfigImpl>())
{
}

ConfigurationReader::ConfigurationReader(const IConfiguration& conf)
    : m_configImpl(std::make_shared<ConfigurationReader::ConfigImpl>())
{
    pt::read_xml(conf.getConfigurationFilename(), m_configImpl->m_configCache, pt::xml_parser::trim_whitespace);
}


ConfigurationReader::ConfigurationReader(const std::string& configuration)
    : m_configImpl(std::make_shared<ConfigurationReader::ConfigImpl>())
{
    std::stringstream ss;
    ss << configuration;
    pt::read_xml(ss, m_configImpl->m_configCache,  pt::xml_parser::trim_whitespace);
}

ConfigurationReader& ConfigurationReader::operator=(const std::string& configuration)
{
    std::stringstream ss;
    ss << configuration;
    pt::read_xml(ss, m_configImpl->m_configCache,  pt::xml_parser::trim_whitespace);
    return *this;
}

double ConfigurationReader::getDoubleParameter(const std::string& parameterName) const
{
    double rv = 0.0;
    try
    {
        rv = m_configImpl->m_configCache.get<double>(parameterName);
    }
    catch(...)
    {

    }
    return rv;
}

std::int32_t ConfigurationReader::getInt32Parameter(const std::string& parameterName) const
{
    std::int32_t rv = -1;
    try
    {
        rv = m_configImpl->m_configCache.get<std::int32_t>(parameterName);
    }
    catch(...)
    {

    }
    return rv;
}

std::string ConfigurationReader::getStringParameter(const std::string& parameterName) const
{
    std::string rv;
    try
    {
        rv = m_configImpl->m_configCache.get<std::string>(parameterName);
    }
    catch(...)
    {

    }
    return rv;
}

std::string ConfigurationReader::getStringAttribute(const std::string& parameterName, const std::string& attributeName) const
{
    std::string rv;
    try
    {
        rv = m_configImpl->m_configCache.get<std::string>(parameterName+".<xmlattr>."+attributeName);
    }
    catch(...)
    {

    }
    return rv;
}

bool ConfigurationReader::getBooleanParameter(const std::string &parameterName) const
{
    bool rv;
    try
    {
        rv = m_configImpl->m_configCache.get<bool>(parameterName);
    }
    catch(...)
    {

    }
    return rv;
}

std::vector<ConfigurationReader> ConfigurationReader::getSiblingsConfiguration(const std::string& siblingsPath, const std::string& siblingName) const
{
    std::vector<ConfigurationReader> result;
    try
    {
        pt::ptree siblings = m_configImpl->m_configCache.get_child(siblingsPath);

        for(const auto& sibling : siblings)
        {
            if(sibling.first == siblingName)
            {
                pt::ptree kconf;
                kconf.add_child(siblingName, sibling.second);
                std::ostringstream oss;
                pt::write_xml(oss, kconf);
                result.emplace_back(oss.str());
            }
        }
    }
    catch(...)
    {

    }
    return result;
}

std::vector<ConfigurationReader> ConfigurationReader::getSiblingsConfiguration(const std::string& siblingsPath, const std::string& siblingName,
                                                                               const std::string& attributeName, const std::string& attributeValue) const
{
    std::vector<ConfigurationReader> result;
    try
    {
        pt::ptree siblings = m_configImpl->m_configCache.get_child(siblingsPath);

        for(const auto& sibling : siblings)
        {
            if(sibling.first == siblingName)
            {
                pt::ptree kconf;
                kconf.add_child(siblingName, sibling.second);
                std::ostringstream oss;
                pt::write_xml(oss, kconf);
                if(kconf.get<std::string>(siblingName+".<xmlattr>."+attributeName) == attributeValue)
                {
                    result.emplace_back(oss.str());
                }
            }
        }
    }
    catch(...)
    {

    }
    return result;
}

ConfigurationReader ConfigurationReader::getSiblingConfiguration(const std::string& siblingsPath, const std::string& siblingName,
                                                                 const std::string& attributeName, const std::string& attributeValue) const
{
    ConfigurationReader result;
    try
    {
        pt::ptree siblings = m_configImpl->m_configCache.get_child(siblingsPath);

        for(const auto& sibling : siblings)
        {
            if(sibling.first == siblingName)
            {
                pt::ptree kconf;
                kconf.add_child(siblingName, sibling.second);
                std::ostringstream oss;
                pt::write_xml(oss, kconf);
                if(kconf.get<std::string>(siblingName+".<xmlattr>."+attributeName) == attributeValue)
                {
                    result = oss.str();
                    break;
                }
            }
        }
    }
    catch(...)
    {

    }
    return result;
}

ConfigurationReader ConfigurationReader::getSiblingConfiguration(const std::string& siblingsPath, const std::string& siblingName,
                                                                 const std::string& attribute1Name, const std::string& attribute1Value,
                                                                 const std::string& attribute2Name, const std::string& attribute2Value) const
{
    ConfigurationReader result;
    try
    {
        pt::ptree siblings = m_configImpl->m_configCache.get_child(siblingsPath);

        for(const auto& sibling : siblings)
        {
            if(sibling.first == siblingName)
            {
                pt::ptree kconf;
                kconf.add_child(siblingName, sibling.second);
                std::ostringstream oss;
                pt::write_xml(oss, kconf);
                if(kconf.get<std::string>(siblingName+".<xmlattr>."+attribute1Name) == attribute1Value &&
                   kconf.get<std::string>(siblingName+".<xmlattr>."+attribute2Name) == attribute2Value)
                {
                    result = oss.str();
                    break;
                }
            }
        }
    }
    catch(...)
    {

    }
    return result;
}
