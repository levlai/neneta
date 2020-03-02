#include <OpenCLContext.h>
#include <ConfigurationReader.h>
#include <boost/log/trivial.hpp>
#include <fstream>

using namespace neneta::gpu;

OpenCLContext::OpenCLContext(const conf::ConfigurationReader &confReader)
    : m_confReader(confReader)
{
    try
    {
        size_t platformId = m_confReader.getInt32Parameter("configuration.gpu.platformid");
        size_t deviceId = m_confReader.getInt32Parameter("configuration.gpu.deviceid");
        std::vector<cl::Platform> platformList;
        cl::Platform::get(&platformList);
        cl_context_properties cprops[3] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platformList[platformId]()), 0};
        if(platformList.size() > platformId)
        {
            std::vector<cl::Device> deviceList;
            platformList[platformId].getDevices(CL_DEVICE_TYPE_ALL, &deviceList);
            if(deviceList.size() > deviceId)
            {
                m_clDevice = deviceList[deviceId];
                BOOST_LOG_TRIVIAL(info) << "Using GPU from " << m_clDevice.getInfo<CL_DEVICE_VENDOR>();
                m_clContext = cl::Context(m_clDevice, cprops, NULL, NULL);
                if(m_confReader.getBooleanParameter("configuration.gpu.profiling"))
                {
                    m_clCommandQueue = cl::CommandQueue(m_clContext, m_clDevice, CL_QUEUE_PROFILING_ENABLE);
                }
                else
                {
                    m_clCommandQueue = cl::CommandQueue(m_clContext, m_clDevice);
                }
            }
        }
    }
    catch(const cl::Error& err)
    {
        BOOST_LOG_TRIVIAL(error) << "GPU::GPU() error: " << err.what();
    }
}


OpenCLContext::~OpenCLContext()
{

}

const cl::Device& OpenCLContext::getDevice() const
{
    return m_clDevice;
}


const cl::Context& OpenCLContext::getContext() const
{
    return m_clContext;
}


const cl::CommandQueue& OpenCLContext::getCommandQueue() const
{
    return m_clCommandQueue;
}


void OpenCLContext::printInfo() const
{
    try
    {
        std::vector<cl::Platform> platformList;
        cl::Platform::get(&platformList);
        for(const auto& platform : platformList)
        {
            std::string desc;
            platform.getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &desc);
            BOOST_LOG_TRIVIAL(info) << "Platform vendor: " << desc;
            platform.getInfo((cl_platform_info)CL_PLATFORM_PROFILE, &desc);
            BOOST_LOG_TRIVIAL(info) << "Platform profile: " << desc;
            platform.getInfo((cl_platform_info)CL_PLATFORM_NAME, &desc);
            BOOST_LOG_TRIVIAL(info) << "Platform name: " << desc;
            platform.getInfo((cl_platform_info)CL_PLATFORM_VERSION, &desc);
            BOOST_LOG_TRIVIAL(info) << "Platform version: " << desc;
            platform.getInfo((cl_platform_info)CL_PLATFORM_EXTENSIONS, &desc);
            BOOST_LOG_TRIVIAL(info) << "Platform extensions: " << desc;

            std::vector<cl::Device> deviceList;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &deviceList);
            for(const auto& device : deviceList)
            {
                BOOST_LOG_TRIVIAL(info) << "Device info:";
                BOOST_LOG_TRIVIAL(info) << "\tName: " << device.getInfo<CL_DEVICE_NAME>();
                BOOST_LOG_TRIVIAL(info) << "\tType: " << device.getInfo<CL_DEVICE_TYPE>();
                BOOST_LOG_TRIVIAL(info) << "\tVendor: " << device.getInfo<CL_DEVICE_VENDOR>();
                BOOST_LOG_TRIVIAL(info) << "\tMax. compute units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
                BOOST_LOG_TRIVIAL(info) << "\tMax. clock freq: " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
                BOOST_LOG_TRIVIAL(info) << "\tMax. mem. alloc. size: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
                BOOST_LOG_TRIVIAL(info) << "\tLocal mem. size: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
                BOOST_LOG_TRIVIAL(info) << "\tGlobal mem. size: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
                BOOST_LOG_TRIVIAL(info) << "\tMax. work group size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
                BOOST_LOG_TRIVIAL(info) << "\tMem. base addr. align: " << device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();
                std::vector<size_t> wi_sizes  = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
                BOOST_LOG_TRIVIAL(info) << "\tMax. work item sizes: " << wi_sizes[0] << ", "  << wi_sizes[1] << ", " << wi_sizes[2];
                BOOST_LOG_TRIVIAL(info) << "\tMax. work item dims.: " << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();

            }
        }
    }
    catch(const cl::Error& err)
    {
        BOOST_LOG_TRIVIAL(error) << "GPU::printInfo() error: " << err.what();
    }

}

