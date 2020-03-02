#include <ImageDispatcher.h>
#include <ConfigurationReader.h>
#include <IImageProcessor.h>
#include <DispatcherEvent.h>
#include <ImageNetDatabase.h>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/log/trivial.hpp>
#include <ThreadPool.h>
#include <map>
#include <unistd.h>
#include <IImageProcessor.h>

namespace bfs = boost::filesystem;
using namespace neneta::imh;

struct ImageDispatcher::ImageProcessorImpl
{
    ImageProcessorImpl(const conf::ConfigurationReader& confReader)
        : m_threadPool(0)
        , m_confReader(confReader)
        , m_runningBatch(BATCH_NOT_AVAILABLE)
        , m_maxQueueSize(confReader.getInt32Parameter("configuration.dispatcher.maxqueuesize"))
        , m_database(std::make_shared<ImageNetDatabase>(confReader))
    {}
    ThreadPool m_threadPool;
    const conf::ConfigurationReader& m_confReader;
    IImageDatabase::BatchId m_runningBatch;
    size_t m_maxQueueSize;
    std::shared_ptr<IImageDatabase> m_database;
    std::map<std::shared_ptr<IImageProcessor>, size_t> m_imageProcessors;
    std::deque<std::shared_ptr<DispatcherEvent>> m_eventQueue;
    std::mutex m_queueMutex;
};

ImageDispatcher::ImageDispatcher(const conf::ConfigurationReader &confReader)
    : m_impl(std::make_shared<ImageProcessorImpl>(confReader))
{
}

ImageDispatcher::~ImageDispatcher()
{
    if(!m_impl->m_threadPool.isStopped())
    {
        m_impl->m_threadPool.shutdown();
    }
}

void ImageDispatcher::registerImageProcessor(std::shared_ptr<IImageProcessor> imgProcessor)
{
    size_t threadId = m_impl->m_threadPool.addThread();
    m_impl->m_imageProcessors.insert({imgProcessor, threadId});
    m_impl->m_threadPool.submit(std::bind(&ImageDispatcher::dispatchImageTask, this, imgProcessor, threadId));
}


void ImageDispatcher::unregisterImageProcessor(std::shared_ptr<IImageProcessor> imgProcessor)
{
    size_t imgProcessorId = m_impl->m_imageProcessors[imgProcessor];
    m_impl->m_threadPool.removeThread(imgProcessorId);
    m_impl->m_imageProcessors.erase(imgProcessor);
}

void ImageDispatcher::wait()
{
    //this one is executed from the process context
    m_impl->m_threadPool.waitForAllTasksToFinish();
    m_impl->m_threadPool.shutdown();
    m_impl->m_database->stopDb();
}

void ImageDispatcher::init()
{   
    m_impl->m_database->startDb();
    m_impl->m_threadPool.addThread();
    m_impl->m_threadPool.addThread();
    m_impl->m_threadPool.submit(std::bind(&ImageDispatcher::updateDbTask, this));
    m_impl->m_threadPool.submit(std::bind(&ImageDispatcher::readImageTask, this));
}

// Dispatcher works on a batch-by-batch, image-by-image basis
// For each processor we have one task
void ImageDispatcher::dispatchImageTask(std::shared_ptr<IImageProcessor> imgProcessor, size_t threadId)
{
    BOOST_LOG_TRIVIAL(debug) << "Started dispatchImageTask.";
    while(!m_impl->m_threadPool.isStopped(threadId))
    {
        std::unique_lock<std::mutex> lock(m_impl->m_queueMutex);
        if(!m_impl->m_eventQueue.empty())
        {
            auto event = m_impl->m_eventQueue.front();
            m_impl->m_eventQueue.pop_front();
            lock.unlock();
            imgProcessor->processTrainingEvent(event);
            if(event->getEventType() == DispatcherEvent::Type::ENDOFDB)
            {
                break;
            }
        }
    }
    BOOST_LOG_TRIVIAL(debug) << "Finished dispatchImageTask.";
}

// There is one task that reads images from the disk and generates coresponding events
void ImageDispatcher::readImageTask()
{
    BOOST_LOG_TRIVIAL(debug) << "Started readImageTask.";
    int sizexy = m_impl->m_confReader.getInt32Parameter("configuration.images.sizexy");
    while(!m_impl->m_threadPool.isStopped())
    {
        std::unique_lock<std::mutex> lock(m_impl->m_queueMutex);
        if(m_impl->m_eventQueue.size() < m_impl->m_maxQueueSize)
        {
            if(m_impl->m_runningBatch == BATCH_NOT_AVAILABLE)
            {
                if((m_impl->m_runningBatch = m_impl->m_database->getNextBatch()) != BATCH_NOT_AVAILABLE)
                {                    
                    m_impl->m_eventQueue.push_back(std::make_shared<StartOfBatchEvent>());
                    BOOST_LOG_TRIVIAL(debug) << "readImageTask() generating StartOfBatchEvent";
                } else
                {
                    if(!m_impl->m_database->empty())
                    {
                        m_impl->m_database->restore();
                        m_impl->m_eventQueue.push_back(std::make_shared<EndOfDbEvent>());
                        BOOST_LOG_TRIVIAL(debug) << "readImageTask() generating EndOfDbEvent";
                        return;
                    }
                    continue;
                }
            }
            else
            {
                std::string imPath;
                if(m_impl->m_database->getNextImage(m_impl->m_runningBatch, imPath))
                {
                    //event queue should contain the image loaded from the disk
                    std::unique_ptr<ImageEvent<cmn::GPUFLOAT>::Image> image = std::unique_ptr<ImageEvent<cmn::GPUFLOAT>::Image>(new ImageEvent<cmn::GPUFLOAT>::Image(imPath));
                    if(image->resize(sizexy))
                    {
                        m_impl->m_eventQueue.push_back(std::make_shared<ImageEvent<cmn::GPUFLOAT>>(std::move(image)));
                        BOOST_LOG_TRIVIAL(debug) << "readImageTask() generating ImageEvent";
                    }
                    else
                    {
                        BOOST_LOG_TRIVIAL(error) << "readImageTask() can't resize " << imPath;
                    }
                }
                else
                {
                    //no more images in the batch
                    m_impl->m_runningBatch = BATCH_NOT_AVAILABLE;
                    m_impl->m_database->setupPersistancePoint();
                    m_impl->m_eventQueue.push_back(std::make_shared<EndOfBatchEvent>());
                    BOOST_LOG_TRIVIAL(debug) << "readImageTask() generating EndOfBatchEvent";
                }
            }
        }
    }
    BOOST_LOG_TRIVIAL(debug) << "Finished readImageTask.";
}

void ImageDispatcher::updateDbTask()
{
    BOOST_LOG_TRIVIAL(debug) << "Starting updateDbTask.";
    bfs::path rootDir(m_impl->m_confReader.getStringParameter("configuration.images.trainset.path"));
    try
    {
        if(bfs::is_directory(rootDir))
        {
            bfs::directory_iterator end;
            int batchId = 0;
            for(bfs::directory_iterator batch(rootDir); batch != end; ++batch)
            {
                if(bfs::is_directory(batch->status()))
                {
                    batchId++;
                    for(auto& image : boost::make_iterator_range(bfs::directory_iterator(batch->path()), {}))
                    {
                        m_impl->m_database->updateImage(batchId, bfs::complete(image).string());
                    }
                    BOOST_LOG_TRIVIAL(debug) << "BatchId " << batchId<< " updated to DB.";
                }
                if(m_impl->m_threadPool.isStopped())
                {
                    //interrupted
                    m_impl->m_database->restoreBatchTable();
                    break;
                }
            }
        }
        else
        {
            BOOST_LOG_TRIVIAL(debug) << "Error in updateDbTask. Check configuration";
        }
    }
    catch(const bfs::filesystem_error& ex)
    {
        BOOST_LOG_TRIVIAL(debug) << "Exception in updateDbTask() :" << ex.what();
    }
    BOOST_LOG_TRIVIAL(debug) << "Finished updateDbTask.";
}
