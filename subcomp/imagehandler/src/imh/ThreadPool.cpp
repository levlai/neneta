#include "ThreadPool.h"
#include <boost/log/trivial.hpp>

using namespace neneta::imh;


ThreadPool::ThreadPool(int numOfThreads)
    : m_stopped(false), m_runningTasksCount(0)
{
    for(int i = 0; i < numOfThreads; i++)
    {       
       addThread();
    }
}

size_t ThreadPool::addThread()
{
    std::unique_lock<std::mutex> lock(m_tasksMutex);
    size_t threadId = m_threadPool.size();
    m_threadPool.insert({threadId, std::make_shared<std::thread>(&ThreadPool::threadMain, this, threadId)});
    return threadId;
}

void ThreadPool::removeThread(size_t threadId)
{
    std::shared_ptr<std::thread> removedThread;
    {
        std::unique_lock<std::mutex> lock(m_tasksMutex);
        if(m_threadPool.count(threadId) > 0)
        {
            removedThread = m_threadPool[threadId];
            m_threadPool.erase(threadId);
        }
    }
    m_condition.notify_all();
    if(removedThread)
    {
        removedThread->join();
    }

}

void ThreadPool::submit(std::function<void()> f)
{
    {
        std::unique_lock<std::mutex> lock(m_tasksMutex);
        m_tasks.push(f);
        m_runningTasksCount++;
    }

    // Wake up one thread.
    m_condition.notify_one();
}

#include <iostream>
void ThreadPool::threadMain(size_t threadId) {

    std::function<void()> task;
    while(true)
    {
        // Scope based locking.
        {
            // Put unique lock on task mutex.
            std::unique_lock<std::mutex> lock(m_tasksMutex);

            // Wait until queue is not empty or termination signal is sent.
            m_condition.wait(lock, [this, threadId]{ return !m_tasks.empty() || (m_threadPool.count(threadId) == 0); });

            // If termination signal received and queue is empty then exit else continue clearing the queue.
            if (m_threadPool.count(threadId) == 0)
            {
                return;
            }

            // Get next task in the queue.
            task = m_tasks.front();

            // Remove it from the queue.
            m_tasks.pop();
        }

        // Execute the task.
        task();
        std::unique_lock<std::mutex> lock(m_tasksMutex);
        m_runningTasksCount--;
        m_emptyTaskQueueCondition.notify_one();
    }
}

void ThreadPool::shutdown()
{
    std::vector<std::shared_ptr<std::thread>> allRunningThreads;
    {
        std::unique_lock<std::mutex> lock(m_tasksMutex);
        for(auto it = m_threadPool.begin(); it != m_threadPool.end(); ++it)
        {
            allRunningThreads.push_back(it->second);
        }
        m_threadPool.clear();
    }

    m_condition.notify_all();

    for(auto& thread : allRunningThreads)
    {
        thread->join();
    }

    m_stopped = true;
}

bool ThreadPool::isStopped() const
{
    return m_stopped;
}

bool ThreadPool::isStopped(size_t threadId)
{
    std::unique_lock<std::mutex> lock(m_tasksMutex);
    return (m_threadPool.count(threadId) == 0);
}

void ThreadPool::waitForAllTasksToFinish()
{
    std::unique_lock<std::mutex> lock(m_tasksMutex);
    m_emptyTaskQueueCondition.wait(lock, [this](){return m_runningTasksCount == 0;});
}

// Destructor.
ThreadPool::~ThreadPool()
{
    if (!isStopped())
    {
        shutdown();
    }
}
