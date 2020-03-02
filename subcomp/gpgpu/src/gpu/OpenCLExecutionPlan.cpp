#include <OpenCLExecutionPlan.h>
#include <IOpenCLChainableExecutionPlan.h>
#include <OpenCLContext.h>
#include <ConfigurationReader.h>
#include <boost/log/trivial.hpp>
#if GCC
#include <cxxabi.h>
#endif

using namespace neneta;
using namespace neneta::gpu;

OpenCLExecutionPlan::OpenCLExecutionPlan(const std::string& id, const conf::ConfigurationReader& confReader, const OpenCLProgram& program)
    : OpenCLBasicExecutionPlan(id)
    , m_confReader(confReader)
    , m_kernelConfiguration(m_confReader)
    , m_program(program)
{
}

OpenCLExecutionPlan::OpenCLExecutionPlan(const OpenCLExecutionPlan& cp)
    : OpenCLBasicExecutionPlan(cp.getId())
    , m_confReader(cp.m_confReader)
    , m_kernelConfiguration(m_confReader)
    , m_program(cp.m_program)
    , m_fpParamSet(cp.m_fpParamSet)
    , m_bpParamSet(cp.m_bpParamSet)
{
}

OpenCLExecutionPlan::~OpenCLExecutionPlan()
{
}

void OpenCLExecutionPlan::planFwd(const boost::variant<std::string, OpenCLExecutionPlan>& task)
{
    if(const std::string* pstr = boost::get<std::string>(&task))
    {
        addKernel(m_fpParamSet, *pstr, {0,0});
    }
    else if(const OpenCLExecutionPlan* pplan = boost::get<OpenCLExecutionPlan>(&task))
    {
        addPlan(m_fpParamSet, *pplan);
    }
}

void OpenCLExecutionPlan::planBck(const boost::variant<std::string, OpenCLExecutionPlan>& task)
{
    if(const std::string* pstr = boost::get<std::string>(&task))
    {
        addKernel(m_bpParamSet, *pstr, {0,0});
    }
    else if(const OpenCLExecutionPlan* pplan = boost::get<OpenCLExecutionPlan>(&task))
    {
        addPlan(m_bpParamSet, *pplan);
    }
}

void OpenCLExecutionPlan::runBckPropagation(const OpenCLContext& context)
{
    m_chainedEvents.clear();
    runBckImpl(context, m_chainedEvents);
}

void OpenCLExecutionPlan::addKernel(PropagationParamSet& pSet, const std::string& kernelName, const OpenCLKernelParameters& params)
{
    try
    {
        const conf::KernelParams& kernelParams = m_kernelConfiguration.getParamSet(kernelName);
        pSet.m_kernels.emplace_back(OpenCLKernel(m_program.getProgram(), kernelName.c_str(), params));
        pSet.m_events.emplace_back();
        pSet.m_profilingHooks.emplace_back(kernelParams.m_profilingEnabled);
    }
    catch(const std::out_of_range& ex)
    {
        BOOST_LOG_TRIVIAL(error) << "Kernel \"" << kernelName << "\" not configured.";
    }
}

void OpenCLExecutionPlan::addPlan(PropagationParamSet& pSet, const OpenCLExecutionPlan& newPlan)
{
    pSet.m_kernels.emplace_back(&newPlan);
    pSet.m_events.emplace_back();
    pSet.m_profilingHooks.emplace_back(false);
}

void OpenCLExecutionPlan::addCopyBufferKernel(PropagationParamSet& pSet, const CopyBufferKernel& newCopyBuffer)
{
    pSet.m_kernels.emplace_back(newCopyBuffer);
    pSet.m_events.emplace_back();
    pSet.m_profilingHooks.emplace_back(false);
}

void OpenCLExecutionPlan::addPlanFwdPropagation(const OpenCLExecutionPlan& newPlan)
{
    m_fpParamSet.m_kernels.emplace_front(&newPlan);
    m_fpParamSet.m_events.emplace_front();
    m_fpParamSet.m_profilingHooks.emplace_front(false);
}

void OpenCLExecutionPlan::addPlanBckPropagation(const OpenCLExecutionPlan& newPlan)
{
    m_bpParamSet.m_kernels.emplace_front(&newPlan);
    m_bpParamSet.m_events.emplace_front();
    m_bpParamSet.m_profilingHooks.emplace_front(false);
}

void OpenCLExecutionPlan::runFwdPropagation(const OpenCLContext& context)
{
    m_chainedEvents.clear();
    runFwdImpl(context, m_chainedEvents);
}

void OpenCLExecutionPlan::runFwdImpl(const OpenCLContext& context, std::vector<cl::Event>& chainedEvents)
{    
    PropagationParamSet& pSet = m_fpParamSet;        
    if(!m_fpParamSet.m_kernels.empty())
    {
        try
        {
            auto end = pSet.m_kernels.end();
            auto kerIt = pSet.m_kernels.begin();
            auto eveIt = pSet.m_events.begin();
            for(kerIt, eveIt; kerIt != end; ++kerIt, ++eveIt)
            {
                switch(kerIt->which())
                {
                case KernelTypes::CL_KERNEL:
                {                   
                    OpenCLKernel& kernel = boost::get<OpenCLKernel>(*kerIt);
                    BOOST_LOG_TRIVIAL(debug) << "Queue kernel " << kernel.getKernel().getInfo<CL_KERNEL_FUNCTION_NAME>();
                    context.getCommandQueue().enqueueNDRangeKernel(kernel.getKernel(), cl::NullRange, kernel.getGWS(), kernel.getLWS(), &chainedEvents, &*eveIt);
                    chainedEvents.push_back(*eveIt);
                    break;
                }
                case KernelTypes::COPY_BUFFER_KERNEL:
                {
                    CopyBufferKernel* pplan = boost::get<CopyBufferKernel>(&*kerIt);
                    assert(pplan->m_to.getInfo<CL_MEM_SIZE>() >= pplan->m_bytesNo || "Invalid m_to buffer size");
                    BOOST_LOG_TRIVIAL(debug) << "Queue copy buffer no. bytes " <<  pplan->m_bytesNo;
                    context.getCommandQueue().enqueueCopyBuffer(pplan->m_from, pplan->m_to, 0, pplan->m_offset, pplan->m_bytesNo, &chainedEvents, &*eveIt);
                    chainedEvents.push_back(*eveIt);
                    break;
                }
                case KernelTypes::EXECUTION_PLAN:
                {
                    BOOST_LOG_TRIVIAL(debug) << "Jump to another execution plan";
                    const OpenCLExecutionPlan* pplan = boost::get<const OpenCLExecutionPlan*>(*kerIt);
                    const_cast<OpenCLExecutionPlan*>(pplan)->runFwdImpl(context, chainedEvents);
                    break;
                }

                }
            }
        }
        catch(cl::Error& err)
        {
            BOOST_LOG_TRIVIAL(error) << "OpenCLExecutionPlan::runImpl() error: " << err.what() << " error no. = " << err.err();
            abort();
        }
    }
}

void OpenCLExecutionPlan::wait() const
{
    cl::WaitForEvents(m_chainedEvents);
}

void OpenCLExecutionPlan::getFwdOrderedProfilingEvents(std::vector<const cl::Event*>& events, std::vector<std::string>& kernelNames) const
{
    auto end = m_fpParamSet.m_kernels.end();
    auto kerIt = m_fpParamSet.m_kernels.begin();
    auto eveIt = m_fpParamSet.m_events.begin();
    auto proIt = m_fpParamSet.m_profilingHooks.begin();
    auto size = m_fpParamSet.m_kernels.size();
    auto i = 0;
    for(i, kerIt, eveIt, proIt; kerIt != end; ++i, ++kerIt, ++eveIt, ++proIt)
    {
        switch(kerIt->which())
        {
        case KernelTypes::CL_KERNEL:
        {
            if(*proIt || i == 0 || i == (size - 1))
            {
                events.emplace_back(&*eveIt);
                kernelNames.emplace_back(boost::get<OpenCLKernel>(*kerIt).getKernelName());
            }
            break;
        }
        case KernelTypes::COPY_BUFFER_KERNEL:
        {
            if(*proIt || i == 0 || i == (size - 1))
            {
                events.emplace_back(&*eveIt);
                kernelNames.emplace_back("copy_buffer_kernel");
            }
            break;
        }
        case KernelTypes::EXECUTION_PLAN:
        {
            boost::get<const OpenCLExecutionPlan*>(*kerIt)->getFwdOrderedProfilingEvents(events, kernelNames);
            break;
        }
        }
    }
}

void OpenCLExecutionPlan::runBckImpl(const OpenCLContext& context, std::vector<cl::Event>& chainedEvents)
{
    PropagationParamSet& pSet = m_bpParamSet;
    if(!pSet.m_kernels.empty())
    {
        try
        {
            auto end = pSet.m_kernels.end();
            auto kerIt = pSet.m_kernels.begin();
            auto eveIt = pSet.m_events.begin();
            for(kerIt, eveIt; kerIt != end; ++kerIt, ++eveIt)
            {
                switch(kerIt->which())
                {
                case KernelTypes::CL_KERNEL:
                {
                    OpenCLKernel& kernel = boost::get<OpenCLKernel>(*kerIt);
                    BOOST_LOG_TRIVIAL(debug) << "Queue kernel " << kernel.getKernel().getInfo<CL_KERNEL_FUNCTION_NAME>();
                    context.getCommandQueue().enqueueNDRangeKernel(kernel.getKernel(), cl::NullRange, kernel.getGWS(), kernel.getLWS(), &chainedEvents, &*eveIt);
                    chainedEvents.push_back(*eveIt);
                    break;
                }
                case KernelTypes::COPY_BUFFER_KERNEL:
                {
                    CopyBufferKernel* pplan = boost::get<CopyBufferKernel>(&*kerIt);
                    assert(pplan->m_to.getInfo<CL_MEM_SIZE>() >= pplan->m_bytesNo || "Invalid m_to buffer size");
                    assert(pplan->m_from.getInfo<CL_MEM_SIZE>() >= pplan->m_bytesNo || "Invalid m_from buffer size");
                    BOOST_LOG_TRIVIAL(debug) << "Queue copy buffer no. bytes " <<  pplan->m_bytesNo;
                    context.getCommandQueue().enqueueCopyBuffer(pplan->m_from, pplan->m_to, 0, pplan->m_offset, pplan->m_bytesNo, &chainedEvents, &*eveIt);
                    chainedEvents.push_back(*eveIt);
                    break;
                }
                case KernelTypes::EXECUTION_PLAN:
                {
                    BOOST_LOG_TRIVIAL(debug) << "Jump to another execution plan";
                    const OpenCLExecutionPlan* pplan = boost::get<const OpenCLExecutionPlan*>(*kerIt);
                    const_cast<OpenCLExecutionPlan*>(pplan)->runBckImpl(context, chainedEvents);
                    break;
                }

                }
            }
        }
        catch(cl::Error& err)
        {
            BOOST_LOG_TRIVIAL(error) << "OpenCLExecutionPlan::runImpl() error: " << err.what() << " error no. = " << err.err();
            abort();
        }
    }
}

void OpenCLExecutionPlan::getBckOrderedProfilingEvents(std::vector<const cl::Event*>& events, std::vector<std::string>& kernelNames) const
{
    auto end = m_fpParamSet.m_kernels.end();
    auto kerIt = m_fpParamSet.m_kernels.begin();
    auto eveIt = m_fpParamSet.m_events.begin();
    auto proIt = m_fpParamSet.m_profilingHooks.begin();
    auto size = m_fpParamSet.m_kernels.size();
    auto i = 0;
    for(i, kerIt, eveIt, proIt; kerIt != end; ++i, ++kerIt, ++eveIt, ++proIt)
    {
        switch(kerIt->which())
        {
        case KernelTypes::CL_KERNEL:
        {
            if(*proIt || i == 0 || i == (size - 1))
            {
                events.emplace_back(&*eveIt);
                kernelNames.emplace_back(boost::get<OpenCLKernel>(*kerIt).getKernelName());
            }
            break;
        }
        case KernelTypes::COPY_BUFFER_KERNEL:
        {
            if(*proIt || i == 0 || i == (size - 1))
            {
                events.emplace_back(&*eveIt);
                kernelNames.emplace_back("copy_buffer_kernel");
            }
            break;
        }
        case KernelTypes::EXECUTION_PLAN:
        {
            boost::get<const OpenCLExecutionPlan*>(*kerIt)->getBckOrderedProfilingEvents(events, kernelNames);
            break;
        }
        }
    }
}

const conf::KernelConfiguration &OpenCLExecutionPlan::getKernelConfiguration() const
{
    return m_kernelConfiguration;
}

void OpenCLExecutionPlan::writeToBuffer(const cl::CommandQueue& queue, const cl::Buffer& buffer, size_t size, void* ptr)
{
    queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, size, ptr);
}

void OpenCLExecutionPlan::writeToBuffer(const cl::CommandQueue& queue, const cl::Buffer& buffer, size_t offset, size_t size, void* ptr)
{
    queue.enqueueWriteBuffer(buffer, CL_TRUE, offset, size, ptr);
}

void OpenCLExecutionPlan::planCopyBufferFwd(const cl::Buffer& from, const cl::Buffer& to, size_t bytesNo)
{
    //add last kernel event from the plan
    addCopyBufferKernel(m_fpParamSet, {from, to, 0, bytesNo});
}

void OpenCLExecutionPlan::planCopyBufferFwd(const cl::Buffer& from, const cl::Buffer& to, size_t offset, size_t bytesNo)
{
    //add last kernel event from the plan
    addCopyBufferKernel(m_fpParamSet, {from, to, offset, bytesNo});
}

void OpenCLExecutionPlan::planCopyBufferBck(const cl::Buffer& from, const cl::Buffer& to, size_t bytesNo)
{
    //add last kernel event from the plan
    addCopyBufferKernel(m_bpParamSet, {from, to, 0, bytesNo});
}

void OpenCLExecutionPlan::planCopyBufferBck(const cl::Buffer& from, const cl::Buffer& to, size_t offset, size_t bytesNo)
{
    //add last kernel event from the plan
    addCopyBufferKernel(m_bpParamSet, {from, to, offset, bytesNo});
}

void OpenCLExecutionPlan::readFromBuffer(const cl::CommandQueue& queue, const cl::Buffer& buffer, size_t size, void* ptr) const
{
    queue.enqueueReadBuffer(buffer, CL_TRUE, 0, size, ptr);
}

OpenCLExecutionPlan& OpenCLExecutionPlan::operator<<(OpenCLExecutionPlan& right)
{    
    IOpenCLChainableExecutionPlan* ptrRight;
    IOpenCLChainableExecutionPlan* ptrLeft;
    if((ptrRight = dynamic_cast<IOpenCLChainableExecutionPlan*>(&right)) && (ptrLeft = dynamic_cast<IOpenCLChainableExecutionPlan*>(this)))
    {
        ptrLeft->setBkpInput(ptrRight->getBkpOutput());
    }
    addPlanBckPropagation(right);
    return right;
}

OpenCLExecutionPlan& OpenCLExecutionPlan::operator>>(OpenCLExecutionPlan& right)
{
    IOpenCLChainableExecutionPlan* ptrRight;
    IOpenCLChainableExecutionPlan* ptrLeft;
    if((ptrRight = dynamic_cast<IOpenCLChainableExecutionPlan*>(&right)) && (ptrLeft = dynamic_cast<IOpenCLChainableExecutionPlan*>(this)))
    {
        ptrRight->setInput(ptrLeft->getOutput());
    }
    right.addPlanFwdPropagation(*this);
    return right;
}

void OpenCLExecutionPlan::printFwdProfilingInfo() const
{
    wait();
    if(m_confReader.getBooleanParameter("configuration.gpu.profiling"))
    {
        try
        {
            std::vector<const cl::Event*> orderedEvents;
            std::vector<std::string> kernelNames;
            getFwdOrderedProfilingEvents(orderedEvents, kernelNames);
            size_t size = orderedEvents.size();
            cl_ulong queued, submit, start, end;
            cl_ulong overallQueued, overallEnd;
            BOOST_LOG_TRIVIAL(debug) << "Execution Plan profiling information:";
            for(size_t i = 0; i < size; ++i)
            {
                BOOST_LOG_TRIVIAL(debug) << "\t\tKernel " << kernelNames[i] << " profiling information:";
                orderedEvents[i]->getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_QUEUED, &queued);
                BOOST_LOG_TRIVIAL(debug) << "\t\t\t\tenqueued to command queue at (OpenCL device time) " << queued * 1e-6 << "ms";
                orderedEvents[i]->getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_SUBMIT, &submit);
                BOOST_LOG_TRIVIAL(debug) << "\t\t\t\tsubmition to the device took " << (submit - queued) * 1e-6 << "ms";
                orderedEvents[i]->getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
                BOOST_LOG_TRIVIAL(debug) << "\t\t\t\tto start execution took " << (start - submit) * 1e-6 << "ms";
                orderedEvents[i]->getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &end);
                BOOST_LOG_TRIVIAL(debug) << "\t\t\t\texecution took " << (end - start) * 1e-6 << "ms";
                BOOST_LOG_TRIVIAL(debug) << "\t\tKernel execution took " << (end - queued) * 1e-6 << "ms.";
                if(i==0)
                {
                    overallQueued = queued;
                }
                else if(i==size-1)
                {
                    overallEnd = end;
                }
            }
            BOOST_LOG_TRIVIAL(debug) << "Execution Plan took " << (overallEnd - overallQueued) * 1e-6 << "ms.";
        }
        catch(cl::Error& err)
        {
            BOOST_LOG_TRIVIAL(error) << "Exception printProfilingInfo error " << err.what() << " no: " << err.err();
        }
    }
    else
    {
        BOOST_LOG_TRIVIAL(info) << "No profiling information available!";
    }
}

void OpenCLExecutionPlan::printBckProfilingInfo() const
{
    wait();
    if(m_confReader.getBooleanParameter("configuration.gpu.profiling"))
    {
        try
        {
            std::vector<const cl::Event*> orderedEvents;
            std::vector<std::string> kernelNames;
            getBckOrderedProfilingEvents(orderedEvents, kernelNames);
            size_t size = orderedEvents.size();
            cl_ulong queued, submit, start, end;
            cl_ulong overallQueued, overallEnd;
            BOOST_LOG_TRIVIAL(debug) << "Execution Plan profiling information:";
            for(size_t i = 0; i < size; ++i)
            {
                BOOST_LOG_TRIVIAL(debug) << "\t\tKernel " << kernelNames[i] << " profiling information:";
                orderedEvents[i]->getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_QUEUED, &queued);
                BOOST_LOG_TRIVIAL(debug) << "\t\t\t\tenqueued to command queue at (OpenCL device time) " << queued * 1e-6 << "ms";
                orderedEvents[i]->getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_SUBMIT, &submit);
                BOOST_LOG_TRIVIAL(debug) << "\t\t\t\tsubmition to the device took " << (submit - queued) * 1e-6 << "ms";
                orderedEvents[i]->getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
                BOOST_LOG_TRIVIAL(debug) << "\t\t\t\tto start execution took " << (start - submit) * 1e-6 << "ms";
                orderedEvents[i]->getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &end);
                BOOST_LOG_TRIVIAL(debug) << "\t\t\t\texecution took " << (end - start) * 1e-6 << "ms";
                BOOST_LOG_TRIVIAL(debug) << "\t\tKernel execution took " << (end - queued) * 1e-6 << "ms.";
                if(i==0)
                {
                    overallQueued = queued;
                }
                else if(i==size-1)
                {
                    overallEnd = end;
                }
            }
            BOOST_LOG_TRIVIAL(debug) << "Execution Plan took " << (overallEnd - overallQueued) * 1e-6 << "ms.";
        }
        catch(cl::Error& err)
        {
            BOOST_LOG_TRIVIAL(error) << "Exception printProfilingInfo error " << err.what() << " no: " << err.err();
        }
    }
    else
    {
        BOOST_LOG_TRIVIAL(info) << "No profiling information available!";
    }
}
