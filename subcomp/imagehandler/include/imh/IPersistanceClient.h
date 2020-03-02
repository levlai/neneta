#pragma once

namespace neneta
{
namespace imh
{

class IImageProcessor;

class IPersistanceClient
{
public:
    virtual void storeDatabase(const std::string& path);
    virtual void restoreDatabase(const std::string& path);
    virtual void storeState(const std::string& path);
    virtual void restoreState(const std::string& path);

protected:
    ~IPersistanceClient();
};

}
}
