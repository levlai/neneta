#include <gtest/gtest.h>
#include "../src/imh/ThreadPool.h"
#include <iostream>
#include <unistd.h>

using namespace neneta::imh;

void f1()
{
    std::cout << "f1()" << std::endl;
}

TEST(ThreadPoolTest, hadamard_product)
{
    ThreadPool tp(0);
    auto id = tp.addThread();
    std::cout << id << std::endl;

    tp.submit(f1);

    tp.submit(f1);
    tp.submit(f1);
    tp.submit(f1);
    tp.submit(f1);
    tp.waitForAllTasksToFinish();
    tp.removeThread(id);
}
