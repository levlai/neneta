#pragma once

#include <string>

namespace neneta
{

namespace gpu
{

class OpenCLBasicExecutionPlan
{
public:
    OpenCLBasicExecutionPlan(const std::string& id) : m_id(id) {}
    virtual ~OpenCLBasicExecutionPlan() {}

    std::string getId() const
    {
        return m_id;
    }

private:
    std::string m_id;
};



}
}
