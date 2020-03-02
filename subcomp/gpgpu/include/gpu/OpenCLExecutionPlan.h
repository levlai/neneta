#pragma once

#include <OpenCLKernel.h>
#include <OpenCLProgram.h>
#include <OpenCLBasicExecutionPlan.h>
#include <IOpenCLChainableExecutionPlan.h>
#include <KernelConfiguration.h>
#include <string>
#include <boost/log/trivial.hpp>
#include <boost/variant.hpp>
#include <list>

namespace neneta
{
namespace conf
{
class ConfigurationReader;
}

namespace gpu
{

class OpenCLProgram;
class OpenCLContext;

struct CopyBufferKernel
{
    CopyBufferKernel(const cl::Buffer& from, const cl::Buffer& to, const size_t offset, const size_t bytesNo)
        : m_from(from)
        , m_to(to)
        , m_offset(offset)
        , m_bytesNo(bytesNo)
    {}


    cl::Buffer m_from;
    cl::Buffer m_to;
    size_t m_offset;
    size_t m_bytesNo;
};

enum KernelTypes
{
    CL_KERNEL,
    EXECUTION_PLAN,
    COPY_BUFFER_KERNEL
};

struct ProfilingFlags
{
    ProfilingFlags() : m_hasStart(false), m_hasEnd(false) {}
    bool m_hasStart;
    bool m_hasEnd;
};

class OpenCLExecutionPlan;

struct PropagationParamSet
{
    PropagationParamSet()
    {
    }
    std::list<boost::variant<OpenCLKernel, const OpenCLExecutionPlan*, CopyBufferKernel>> m_kernels;
    std::list<cl::Event> m_events;
    std::list<std::uint8_t> m_profilingHooks;
};

class OpenCLExecutionPlan : public OpenCLBasicExecutionPlan
{
public:
    OpenCLExecutionPlan(const std::string& id, const conf::ConfigurationReader& confReader, const OpenCLProgram& program);
    OpenCLExecutionPlan(const OpenCLExecutionPlan& cp);
    OpenCLExecutionPlan& operator=(const OpenCLExecutionPlan& cp) = delete;
    virtual ~OpenCLExecutionPlan();

    template<typename ...Args>
    void planFwd(const boost::variant<std::string, OpenCLExecutionPlan>& task, Args&&... args)
    {
        if(const std::string* pstr = boost::get<std::string>(&task))
        {
            addKernel(m_fpParamSet, *pstr, std::forward<Args>(args)...);
        }
        else if(const OpenCLExecutionPlan* pplan = boost::get<OpenCLExecutionPlan>(&task))
        {
            addPlan(m_fpParamSet, *pplan);
        }
    }
    void planFwd(const boost::variant<std::string, OpenCLExecutionPlan>& task);
    void planCopyBufferFwd(const cl::Buffer& from, const cl::Buffer& to, size_t bytesNo);
    void planCopyBufferFwd(const cl::Buffer& from, const cl::Buffer& to, size_t offset, size_t bytesNo);

    template<typename ...Args>
    void planBck(const boost::variant<std::string, OpenCLExecutionPlan>& task, Args&&... args)
    {
        if(const std::string* pstr = boost::get<std::string>(&task))
        {
            addKernel(m_bpParamSet, *pstr, std::forward<Args>(args)...);
        }
        else if(const OpenCLExecutionPlan* pplan = boost::get<OpenCLExecutionPlan>(&task))
        {
            addPlan(m_bpParamSet, *pplan);
        }
    }
    void planBck(const boost::variant<std::string, OpenCLExecutionPlan>& task);
    void planCopyBufferBck(const cl::Buffer& from, const cl::Buffer& to, size_t bytesNo);
    void planCopyBufferBck(const cl::Buffer& from, const cl::Buffer& to, size_t offset, size_t bytesNo);

    void runFwdPropagation(const OpenCLContext& context);
    void runBckPropagation(const OpenCLContext& context);

    void wait() const;
    virtual void printFwdProfilingInfo() const;
    virtual void printBckProfilingInfo() const;

    const conf::KernelConfiguration& getKernelConfiguration() const;

    void readFromBuffer(const cl::CommandQueue& queue, const cl::Buffer& buffer, size_t size, void* ptr) const;
    void writeToBuffer(const cl::CommandQueue& queue, const cl::Buffer& buffer, size_t size, void* ptr);
    void writeToBuffer(const cl::CommandQueue& queue, const cl::Buffer& buffer, size_t offset, size_t size, void* ptr);
    template<typename FillType>
    void fillBuffer(const cl::CommandQueue& queue, const cl::Buffer& buffer, unsigned int size, const FillType value)
    {
        queue.enqueueFillBuffer(buffer, value, 0, size);
    }

    OpenCLExecutionPlan& operator<<(OpenCLExecutionPlan& left);
    OpenCLExecutionPlan& operator>>(OpenCLExecutionPlan& right);

private:    
    template<typename Arg, typename ...Rest>
    void setArgs(int count, cl::Kernel& kernel, Arg&& arg, Rest&&... tail)
    {
        try
        {
            kernel.setArg(count, std::forward<Arg>(arg));
        }
        catch(const cl::Error& err)
        {
            BOOST_LOG_TRIVIAL(debug) << "Setting arg " << count << " rc = " << err.err();
        }
        setArgs(++count, kernel, std::forward<Rest>(tail)...);
    }

    template<typename Arg, typename ...Rest>
    void setArgs(int count, cl::Kernel& kernel, Arg&& arg)
    {
        try
        {
            kernel.setArg(count, std::forward<Arg>(arg));
        }
        catch(const cl::Error& err)
        {
            BOOST_LOG_TRIVIAL(debug) << "Setting last arg " << count << " rc = " << err.err();
        }
    }

    template<typename KPs, typename ...Args>
    void addKernel(PropagationParamSet& pSet, const std::string& kernelName, KPs&& kernelParams, Args&&... args)
    {
        try
        {           
          //  BOOST_LOG_TRIVIAL(debug) << "Adding kernel \"" << kernelName;
            pSet.m_kernels.emplace_back(OpenCLKernel(m_program.getProgram(), kernelName.c_str(), std::forward<KPs>(kernelParams)));
            const conf::KernelParams& kernelParams = m_kernelConfiguration.getParamSet(kernelName);
            pSet.m_events.emplace_back();
            pSet.m_profilingHooks.emplace_back(kernelParams.m_profilingEnabled);
            setArgs(0, boost::get<OpenCLKernel>(pSet.m_kernels.back()).getKernel(), std::forward<Args>(args)...);
        }
        catch(const cl::Error& err)
        {
            BOOST_LOG_TRIVIAL(error) << "Exception in OpenCLExecutionPlan::plan() for kernel " << kernelName << " - " << err.what() << " - ERROR = " << err.err();
        }
        catch(const std::out_of_range& ex)
        {
            BOOST_LOG_TRIVIAL(error) << "Kernel \"" << kernelName << "\" not configured.";
        }
    }
    void addKernel(PropagationParamSet& pSet, const std::string& kernelName, const OpenCLKernelParameters& params);
    void addPlan(PropagationParamSet& pSet, const OpenCLExecutionPlan& newPlan);
    void addCopyBufferKernel(PropagationParamSet& pSet, const CopyBufferKernel& newCopyBuffer);

    //forward propagation
    void addPlanFwdPropagation(const OpenCLExecutionPlan& newPlan);
    void runFwdImpl(const OpenCLContext& context, std::vector<cl::Event>& chainedEvents);    
    void getFwdOrderedProfilingEvents(std::vector<const cl::Event*>& events, std::vector<std::string>& kernelNames) const;

    //back propagation
    void addPlanBckPropagation(const OpenCLExecutionPlan& newPlan);
    void runBckImpl(const OpenCLContext& context, std::vector<cl::Event>& chainedEvents);    
    void getBckOrderedProfilingEvents(std::vector<const cl::Event*>& events, std::vector<std::string>& kernelNames) const;

private:
    const conf::ConfigurationReader& m_confReader;
    conf::KernelConfiguration m_kernelConfiguration;
    const OpenCLProgram& m_program;   
    std::vector<cl::Event> m_chainedEvents;
    PropagationParamSet m_fpParamSet;
    PropagationParamSet m_bpParamSet;
};

}
}
