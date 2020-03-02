#include <OpenCLContext.h>
#include <OpenCLProgram.h>
#include <ConfigurationReader.h>
#include <boost/log/trivial.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <fstream>

namespace bfs = boost::filesystem;
using namespace neneta::gpu;

OpenCLProgram::OpenCLProgram(const conf::ConfigurationReader& confReader)
    : m_confReader(confReader)
{
    if(!m_confReader.getStringParameter("configuration.gpu.kernels").empty())
    {
        addSource(m_confReader.getStringParameter("configuration.gpu.kernels"));
    }
    else
    {
        addSources();
    }
}

OpenCLProgram::~OpenCLProgram()
{

}

void OpenCLProgram::addSource(const std::string& programSourceFilename)
{
    std::ifstream sourceFile(programSourceFilename);
    if(sourceFile.is_open())
    {
        BOOST_LOG_TRIVIAL(debug) << "OpenCLProgram::addSource() adding new *.cl file : " << programSourceFilename;
        std::string prog{std::istreambuf_iterator<char>{sourceFile}, std::istreambuf_iterator<char>{}};
        m_clProgramSources.push_back(prog);
    }

}

void OpenCLProgram::addSources()
{
    bfs::path rootDir(bfs::current_path());
    if(!m_confReader.getStringParameter("configuration.gpu.sourcesdir").empty())
    {
        rootDir = bfs::system_complete(m_confReader.getStringParameter("configuration.gpu.sourcesdir"));
    }
    try
    {
        if(bfs::is_directory(rootDir))
        {
            bfs::directory_iterator end;
            for(auto& filename : boost::make_iterator_range(bfs::directory_iterator(rootDir), {}))
            {
                if(!bfs::is_regular_file(filename.status()) ||
                        bfs::complete(filename).string().substr(bfs::complete(filename).string().length() - 3).compare(".cl") != 0) continue;
                addSource(bfs::complete(filename).string());
            }
        }
        else if(bfs::is_regular_file(rootDir))
        {
            addSource(bfs::complete(rootDir).string());
        }
    }
    catch(const bfs::filesystem_error& ex)
    {
        BOOST_LOG_TRIVIAL(error) << "Exception in OpenCLProgram::addSources() : " << ex.what();
    }
}

bool OpenCLProgram::compile(const OpenCLContext& openCLContext)
{
    bool success = false;
    try
    {
        cl::Program::Sources sources;
        for(const std::string& source : m_clProgramSources)
        {
            sources.emplace_back(source.c_str(), source.size() + 1);
        }
        m_clProgram = cl::Program(openCLContext.getContext(), sources);
        m_clProgram.build({openCLContext.getDevice()}, "-I./ -cl-std=CL1.2 -g");
        success = true;
        BOOST_LOG_TRIVIAL(debug) << "Compilation log: \n" << m_clProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(openCLContext.getDevice());
    }
    catch(const cl::Error& err)
    {
        BOOST_LOG_TRIVIAL(error) << "OpenCLProgram::compile() error: " << err.what() << " id " << err.err();
        BOOST_LOG_TRIVIAL(error) << "Compilation error log: \n" << m_clProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(openCLContext.getDevice());
    }
    return success;

}

const cl::Program& OpenCLProgram::getProgram() const
{
    return m_clProgram;
}
