#pragma once

#include <IConfiguration.h>
#include <string>
#include <map>

namespace neneta
{
namespace conf
{

template<typename ParamSet>
class Configuration : public IConfiguration
{
public:
    typedef std::map<std::string, ParamSet> ParamSetsMap;

    Configuration(const std::string& configFilename) : m_configFilename(configFilename) {}
    ~Configuration() {}

    std::string getConfigurationFilename() const
    {
        return m_configFilename;
    }

    template<typename... Args>
    void updateParamSetsMap(Args&&... args)
    {
        m_paramSetsMap.emplace(std::forward<Args>(args)...);
    }

    const ParamSetsMap& getParamSetsMap() const
    {
        return m_paramSetsMap;
    }

    const ParamSet& getParamSet(const std::string& id) const
    {
        return m_paramSetsMap.at(id.c_str());
    }

private:
    std::string m_configFilename;
    ParamSetsMap m_paramSetsMap;
};

}
}
