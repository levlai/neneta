project(neuralnetworktests)
cmake_minimum_required(VERSION 2.8)


add_executable(${PROJECT_NAME}
    NeuralNetworkTests.cpp
    FCLayerTests.cpp
    SoftMaxLayerTests.cpp
    ConvLayerTests.cpp
    PoolingLayerTests.cpp
    BackpropagationTest.cpp
    ProjectionLayerTests.cpp
    ConvLayerBpTests.cpp
    ConvLayerBpMCTest.cpp
    TestMain.cpp        
)

target_link_libraries(${PROJECT_NAME}
    neuralnetwork
    ${GTEST_LIB}
    ${GMOCK_LIB}
    imagehandler
    plotting
    ${XLIB}
    common
)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
