#pragma once

#include <string>
#include <memory>
#include <vector>
#include <Types.h>

namespace neneta
{
namespace conf
{
class ConfigurationReader;
}

namespace pers
{

class Persistance
{
public:
    Persistance(const conf::ConfigurationReader& confReader);

    void storeFloatBlob(const std::string& id, const std::vector<cmn::GPUFLOAT>& blob);
    void restoreFloatBlob(const std::string& id, std::vector<cmn::GPUFLOAT>& blob) const;    


private:
    class PersistanceImpl;
    std::shared_ptr<PersistanceImpl> m_impl;
    bool m_storeActive;
    bool m_restoreActive;
};

}
}


/*
 *
 * Life cycle of neneta (./neneta --persistance=/path/to/pers/data/2016-03-05_2254 --snapshot=1 ... or ./neneta --config=environment.xml
 * 1. loading images (storing the batch list) <batch_id, class_id, image_path>
 * 3. creating network based on configuration (storing the initialized weights)
 * 4. during runtime (storing the snapshot)
 *
 * Snapshot:
 * 1. snapshot is taken each time batch is processed (configurable)
 * 2. information: network weights, and next batch to be executed is saved
 *
 *
 * Persistance should provide an interface of where the data should be stored.
 * For example   storePathForCon
 *               getPathOfStored ?
 */
