#include <gtest/gtest.h>
#include <ImageNetDatabase.h>
#include <ConfigurationReader.h>
#include <IConfiguration.h>
#include <iostream>

using namespace neneta;
using namespace neneta::imh;

TEST(ImangeNetDatabase, db_tests)
{
    conf::ConfigurationReader envReader("configuration_test.xml");

    ImageNetDatabase db(envReader);
    auto batch = db.getNextBatch();
    std::string path;
    db.getNextImage(batch, path);
    std::cout << path << std::endl;
    db.getNextImage(batch+1, path);
    std::cout << path << std::endl;
}
