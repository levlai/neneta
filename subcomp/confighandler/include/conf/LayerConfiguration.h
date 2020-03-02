#pragma once

namespace neneta
{
namespace conf
{

enum ELayerType
{
    INPUT,
    FOURIER,
    SPECTRAL_CONV,
    SPECTRAL_POOL,
    FULLY_CONN,
    SOFT_MAX,
    ERROR_CALC,
    PROJECTION
};

struct LayerParams
{
    LayerParams(ELayerType type)
        : m_type(type)
    {}

    ELayerType getType() const
    {
        return m_type;
    }

    ELayerType   m_type;
};


}
}
