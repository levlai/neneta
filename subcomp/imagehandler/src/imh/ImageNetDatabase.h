#pragma once

#include <IImageDatabase.h>
#include <memory>

namespace neneta
{
namespace conf
{
class ConfigurationReader;
}

namespace imh
{

class ImageNetDatabase : public IImageDatabase
{
public:
    ImageNetDatabase(const conf::ConfigurationReader& confReader);
    ~ImageNetDatabase();

    void startDb();
    void stopDb();
    BatchId getNextBatch() const;
    bool getNextImage(BatchId batch, Path& path) const;
    void updateImage(const BatchId& batch, const Path& path);
    void restoreBatchTable();
    bool empty() const;
    void restore();
    void setupPersistancePoint();

private:
    //statements
    class CashedData;
    std::shared_ptr<CashedData> m_database;
};

}
}

