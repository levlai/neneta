#pragma once

namespace neneta
{
namespace net
{

class IPersistedLayer
{
public:    
    virtual void store() = 0;
    virtual void restore() = 0;

protected:
    ~IPersistedLayer() {}
};

} // net
} // neneta
