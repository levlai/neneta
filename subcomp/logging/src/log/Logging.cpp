#include <Logging.h>
#include <ConfigurationReader.h>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>

namespace bl = boost::log;
namespace keywords = boost::log::keywords;
namespace bfs = boost::filesystem;
using namespace neneta::logging;

Logging::Logging(const conf::ConfigurationReader& confReader)
    : m_confReader(confReader)
{
}

void Logging::init()
{
    bfs::path fileName(m_confReader.getStringParameter("configuration.logging.filepath"));
    fileName /= m_confReader.getStringParameter("configuration.logging.filename");
    bl::add_file_log
    (
        keywords::file_name = fileName.string(),
        keywords::rotation_size = m_confReader.getInt32Parameter("configuration.logging.rotationsize") * 1024 * 1024,
        keywords::format = m_confReader.getStringParameter("configuration.logging.format"),
        keywords::auto_flush = true
    );

    std::int32_t severity = m_confReader.getInt32Parameter("configuration.logging.level");
    switch(severity)
    {
    case 0:
        bl::core::get()->set_filter(bl::trivial::severity >= bl::trivial::trace);
        break;
    case 1:
        bl::core::get()->set_filter(bl::trivial::severity >= bl::trivial::debug);
        break;
    case 2:
        bl::core::get()->set_filter(bl::trivial::severity >= bl::trivial::info);
        break;
    case 3:
        bl::core::get()->set_filter(bl::trivial::severity >= bl::trivial::warning);
        break;
    case 4:
        bl::core::get()->set_filter(bl::trivial::severity >= bl::trivial::error);
        break;
    case 5:
        bl::core::get()->set_filter(bl::trivial::severity >= bl::trivial::fatal);
        break;
    default:
        bl::core::get()->set_filter(bl::trivial::severity >= bl::trivial::debug);
    }

    bl::add_common_attributes();
}
