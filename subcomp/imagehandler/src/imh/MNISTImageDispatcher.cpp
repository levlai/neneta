#include <MNISTImageDispatcher.h>
#include <ConfigurationReader.h>
#include <IImageProcessor.h>
#include <DispatcherEvent.h>
#include <boost/log/trivial.hpp>
#include <fstream>
#include <ThreadPool.h>
#include <map>
#include <deque>

using namespace neneta::imh;

std::map<MNISTImageDispatcher::DispatchType, std::string> DispatchTypeResolver =
{
    {MNISTImageDispatcher::DispatchType::TRAINING, "trainset"},
    {MNISTImageDispatcher::DispatchType::VALIDATION, "validationset"},
    {MNISTImageDispatcher::DispatchType::TEST, "testset"},
};

struct MNISTImageDispatcher::ImageProcessorImpl
{
    ImageProcessorImpl(DispatchType type, const conf::ConfigurationReader& confReader)
        : m_type(type)
        , m_imgsPath(confReader.getStringParameter(std::string("configuration.images.")+DispatchTypeResolver[type]+std::string(".path")))
        , m_lblsPath(confReader.getStringParameter(std::string("configuration.images.")+DispatchTypeResolver[type]+std::string(".labels")))
        , m_batchSize((type==MNISTImageDispatcher::DispatchType::TRAINING)?confReader.getInt32Parameter("configuration.images.trainset.minibatchsize"):1)
        , m_offset(confReader.getInt32Parameter(std::string("configuration.images.")+DispatchTypeResolver[type]+std::string(".offset")))
        , m_size(confReader.getInt32Parameter(std::string("configuration.images.")+DispatchTypeResolver[type]+std::string(".size")))
        , m_threadPool(0)
    {        
        BOOST_LOG_TRIVIAL(info) << "Batch size = " << m_batchSize;
    }

    MNISTImageDispatcher::DispatchType m_type;
    std::string m_imgsPath;
    std::string m_lblsPath;
    unsigned int m_batchSize;    
    unsigned int m_offset;
    unsigned int m_size;
    ThreadPool m_threadPool;       
    std::int32_t m_numOfImages;    
    std::int32_t m_numOfRows;
    std::int32_t m_numOfCols;    
    std::map<std::shared_ptr<IImageProcessor>, size_t> m_imageProcessors;
    std::deque<std::shared_ptr<DispatcherEvent>> m_eventQueue;
    std::mutex m_queueMutex;
};

std::int32_t swapBytes(std::int32_t num)
{
    return ((num>>24)&0x000000FF) | ((num>>8)&0x0000FF00) | ((num<<8)&0x00FF0000) | ((num<<24)&0xFF000000);
}

MNISTImageDispatcher::MNISTImageDispatcher(DispatchType type, const conf::ConfigurationReader &confReader)
    : m_impl(std::make_shared<ImageProcessorImpl>(type, confReader))
{
}

MNISTImageDispatcher::~MNISTImageDispatcher()
{
    if(!m_impl->m_threadPool.isStopped())
    {
        m_impl->m_threadPool.shutdown();
    }
}

void MNISTImageDispatcher::registerImageProcessor(std::shared_ptr<IImageProcessor> imgProcessor)
{
    size_t threadId = m_impl->m_threadPool.addThread();
    m_impl->m_imageProcessors.insert({imgProcessor, threadId});
    m_impl->m_threadPool.submit(std::bind(&MNISTImageDispatcher::dispatchImageTask, this, imgProcessor, threadId));
}


void MNISTImageDispatcher::unregisterImageProcessor(std::shared_ptr<IImageProcessor> imgProcessor)
{
    size_t imgProcessorId = m_impl->m_imageProcessors[imgProcessor];
    m_impl->m_threadPool.removeThread(imgProcessorId);
    m_impl->m_imageProcessors.erase(imgProcessor);
}

void MNISTImageDispatcher::wait()
{
    m_impl->m_threadPool.waitForAllTasksToFinish();
    m_impl->m_threadPool.shutdown();
}

void MNISTImageDispatcher::init()
{   
    m_impl->m_threadPool.addThread();
    m_impl->m_threadPool.submit(std::bind(&MNISTImageDispatcher::readImageTask, this));
}

void MNISTImageDispatcher::dispatchImageTask(std::shared_ptr<IImageProcessor> imgProcessor, size_t threadId)
{
    BOOST_LOG_TRIVIAL(info) << "Started dispatchImageTask.";
    while(!m_impl->m_threadPool.isStopped(threadId))
    {
        std::unique_lock<std::mutex> lock(m_impl->m_queueMutex);        
        if(!m_impl->m_eventQueue.empty())
        {
            auto event = m_impl->m_eventQueue.front();
            m_impl->m_eventQueue.pop_front();
            lock.unlock();
            switch(m_impl->m_type)
            {
                case DispatchType::TRAINING:
                    imgProcessor->processTrainingEvent(event);
                    break;
                case DispatchType::VALIDATION:
                    imgProcessor->processValidationEvent(event);
                    break;
                case DispatchType::TEST:
                    imgProcessor->processTestEvent(event);
                    break;
            }
            if(event->getEventType() == DispatcherEvent::Type::ENDOFDB)
            {
                break;
            }
        }        
    }
    BOOST_LOG_TRIVIAL(info) << "Finished dispatchImageTask.";
}

void MNISTImageDispatcher::readImageTask()
{
    BOOST_LOG_TRIVIAL(info) << "Started readImageTask.";
    std::ifstream imagesfile(m_impl->m_imgsPath, std::ios::binary);
    std::ifstream labelsfile(m_impl->m_lblsPath, std::ios::binary);
    std::streampos startOfImgs = readTrainingImagesFile(imagesfile);
    std::streampos startOfLbls = readTrainingLabelsFile(labelsfile);
    startOfImgs += std::streampos(m_impl->m_offset*m_impl->m_numOfRows*m_impl->m_numOfCols);
    startOfLbls += std::streampos(m_impl->m_offset);
    imagesfile.seekg(startOfImgs);
    labelsfile.seekg(startOfLbls);
    readImages(imagesfile, labelsfile);
    BOOST_LOG_TRIVIAL(info) << "Finished readImageTask.";
}

std::streampos MNISTImageDispatcher::readTrainingImagesFile(std::ifstream& imagesfile)
{    
    std::streampos res = 0;
    if(imagesfile)
    {
        std::int32_t tmp;
        imagesfile.read(reinterpret_cast<char*>(&tmp), sizeof(std::int32_t));
        BOOST_LOG_TRIVIAL(info) << "MNIST Image file magic = " << swapBytes(tmp);
        assert(swapBytes(tmp) == 2051);

        imagesfile.read(reinterpret_cast<char*>(&tmp), sizeof(std::int32_t));
        BOOST_LOG_TRIVIAL(info) << "MNIST num of images  = " << swapBytes(tmp);
        m_impl->m_numOfImages = swapBytes(tmp);
        assert(m_impl->m_numOfImages > 0);

        imagesfile.read(reinterpret_cast<char*>(&tmp), sizeof(std::int32_t));
        BOOST_LOG_TRIVIAL(info) << "MNIST num of rows  = " << swapBytes(tmp);
        m_impl->m_numOfRows = swapBytes(tmp);
        assert(m_impl->m_numOfRows > 0);

        imagesfile.read(reinterpret_cast<char*>(&tmp), sizeof(std::int32_t));
        BOOST_LOG_TRIVIAL(info) << "MNIST num of cols  = " << swapBytes(tmp);
        m_impl->m_numOfCols = swapBytes(tmp);
        assert(m_impl->m_numOfCols > 0);

        res = imagesfile.tellg();
    }
    return res;
}


std::streampos MNISTImageDispatcher::readTrainingLabelsFile(std::ifstream& labelsfile)
{    
    std::streampos res = 0;
    if(labelsfile)
    {
        std::int32_t tmp;
        labelsfile.read(reinterpret_cast<char*>(&tmp), sizeof(std::int32_t));
        BOOST_LOG_TRIVIAL(info) << "MNIST Labels file magic = " << swapBytes(tmp);
        assert(swapBytes(tmp) == 2049);

        labelsfile.read(reinterpret_cast<char*>(&tmp), sizeof(std::int32_t));
        BOOST_LOG_TRIVIAL(info) << "MNIST num of labels  = " << swapBytes(tmp);

        res = labelsfile.tellg();
    }
    return res;
}

void MNISTImageDispatcher::readImages(std::ifstream& imagesfile, std::ifstream& labelsfile)
{
    std::unique_ptr<std::uint8_t> buffer = std::unique_ptr<std::uint8_t>(new std::uint8_t[m_impl->m_numOfCols*m_impl->m_numOfRows]);
    std::uint8_t label;

    {
        std::unique_lock<std::mutex> lock(m_impl->m_queueMutex);
        m_impl->m_eventQueue.emplace_back(std::make_shared<StartOfDbEvent>());
    }

    for(int i = 0; i < m_impl->m_size; ++i)
    {
        std::unique_lock<std::mutex> lock(m_impl->m_queueMutex);
        if(i%m_impl->m_batchSize == 0)
        {
            m_impl->m_eventQueue.emplace_back(std::make_shared<StartOfBatchEvent>());
        }

        labelsfile.read(reinterpret_cast<char*>(&label), sizeof(std::uint8_t));
        imagesfile.read(reinterpret_cast<char*>(buffer.get()), m_impl->m_numOfCols*m_impl->m_numOfRows);
        std::unique_ptr<ip::Image<std::uint8_t, std::int16_t>> image = std::unique_ptr<ip::Image<std::uint8_t, std::int16_t>>(new ip::Image<std::uint8_t, std::int16_t>(buffer.get(), m_impl->m_numOfCols, m_impl->m_numOfRows, label));
        m_impl->m_eventQueue.emplace_back(std::make_shared<ImageEvent<std::uint8_t>>(std::move(image)));

        if((i < m_impl->m_size - 1) && ((i+1)%m_impl->m_batchSize == 0))
        {
           m_impl->m_eventQueue.emplace_back(std::make_shared<EndOfBatchEvent>());
        }
    }

    std::unique_lock<std::mutex> lock(m_impl->m_queueMutex);
    m_impl->m_eventQueue.emplace_back(std::make_shared<EndOfDbEvent>());
}

