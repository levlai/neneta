#pragma once

#include <functional>
#include <thread>
#include <map>
#include <set>
#include <queue>
#include <mutex>
#include <list>
#include <condition_variable>

namespace neneta
{
namespace imh
{

class ThreadPool
{
public:
    ThreadPool(int numOfThreads);
    ~ThreadPool();

    size_t addThread();
    void removeThread(size_t threadId);
    void submit(std::function<void()> f);
    void shutdown();
    bool isStopped() const;
    bool isStopped(size_t threadId);
    void waitForAllTasksToFinish();

private:
    // Function that will be invoked by our threads.
    void threadMain(size_t id);

private:
    std::map<size_t, std::shared_ptr<std::thread>> m_threadPool;
    std::list<std::thread> m_removedThreads;
    std::queue<std::function<void()>> m_tasks;    
    std::mutex m_tasksMutex;
    std::condition_variable m_condition;
    std::condition_variable m_emptyTaskQueueCondition;

    // Indicates that pool has been terminated.
    bool m_stopped;
    int m_runningTasksCount;
};

}
}
