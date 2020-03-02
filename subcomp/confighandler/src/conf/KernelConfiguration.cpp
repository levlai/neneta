#include <KernelConfiguration.h>
#include <ConfigurationReader.h>

using namespace neneta::conf;

KernelConfiguration::KernelConfiguration(const ConfigurationReader& confReader)
    : Configuration(confReader.getStringParameter("configuration.gpu.kernelconfig"))
{
    ConfigurationReader kernelsConfigReader(*this);
    for(const auto& kernelConf : kernelsConfigReader.getSiblingsConfiguration("kernels", "kernel"))
    {
         updateParamSetsMap(std::piecewise_construct,
                           std::forward_as_tuple(kernelConf.getStringAttribute("kernel", "name")),
                           std::forward_as_tuple(kernelConf.getBooleanParameter("kernel.profile")));
    }
    m_lws1 = kernelsConfigReader.getInt32Parameter("kernels.localworksize.dim1") ;
    m_lws2 = kernelsConfigReader.getInt32Parameter("kernels.localworksize.dim2") ;
    m_lws3 = kernelsConfigReader.getInt32Parameter("kernels.localworksize.dim3") ;
}

unsigned int KernelConfiguration::getLWS(unsigned int dim) const
{
    unsigned int res = 0;
    switch(dim)
    {
    case 1:
        res = m_lws1;
        break;
    case 2:
        res = m_lws2;
        break;
    case 3:
        res = m_lws3;
        break;
    }
    return res;
}
