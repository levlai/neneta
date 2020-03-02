#pragma once

#include <string>

namespace neneta
{
namespace imh
{

class IImageDatabase
{
public:
    typedef int BatchId;
    typedef std::string Path;

    virtual void startDb() = 0;
    virtual void stopDb() = 0;
    virtual BatchId getNextBatch() const = 0;
    virtual bool getNextImage(BatchId batch, Path& path) const = 0;
    virtual void updateImage(const BatchId& batch, const Path& path) = 0;
    virtual void restoreBatchTable() = 0;
    virtual bool empty() const = 0;
    virtual void restore() = 0;
    virtual void setupPersistancePoint() = 0;

    virtual ~IImageDatabase() {}
};

const IImageDatabase::BatchId BATCH_NOT_AVAILABLE = 0;

}
}

