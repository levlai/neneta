#pragma once

#include <vector>
#include <list>

namespace cl
{
class Buffer;
}

namespace neneta
{

namespace gpu
{

typedef cl::Buffer BufferType;

struct SharedBuffers        
{    
    void clear()
    {
        m_reChannels.clear();
        m_imChannels.clear();
    }

    //not always needed, only aliases
    std::vector<BufferType> m_reChannels;
    std::vector<BufferType> m_imChannels;
    //filled always:
    BufferType m_reShMem;
    BufferType m_imShMem;
    //backup
    BufferType m_reShMemBkp;
    BufferType m_imShMemBkp;
    //desired output
    BufferType m_reDesired;
    BufferType m_imDesired;
};

typedef SharedBuffers BufferIO;

struct LayerConnectionInfo
{
    enum Type
    {
        FULLY_CONNECTED,
        CONVOLUTIONAL
    };

    LayerConnectionInfo() : m_numOfNeurons(0), m_numOfConnectionsPerNeuron(0), m_stride(1),  m_kernelSize(0), m_deltaSize(0), m_type(Type::FULLY_CONNECTED) {}
    unsigned int m_numOfNeurons;
    unsigned int m_numOfConnectionsPerNeuron;
    unsigned int m_stride;
    unsigned int m_kernelSize;
    unsigned int m_deltaSize;
    Type m_type;
};

class IOpenCLChainableExecutionPlan
{
public:    
    virtual void setInput(BufferIO input) = 0;
    virtual BufferIO getOutput() = 0;

    virtual void setBkpInput(BufferIO input) = 0;
    virtual BufferIO getBkpOutput() = 0;

protected:
    ~IOpenCLChainableExecutionPlan() {}
};


} // gpu
} // neneta
