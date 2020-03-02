#include <gmock/gmock.h>
#include <gtest/gtest.h>

int main(int argc, char* argv[])
{
    testing::InitGoogleMock(&argc, argv);
    int success = RUN_ALL_TESTS();

    return success;
}

