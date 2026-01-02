#include "gtest/gtest.h"

TEST(gtest, basic)
{
    std::cout << "Hello, Google Test!!" << std::endl;
    EXPECT_GT(3, 0);
    EXPECT_EQ(2, 2);
}