project(imh_test)
cmake_minimum_required(VERSION 2.8)

file(GLOB_RECURSE INC_ALL "*.h")

add_executable(${PROJECT_NAME}
    ImageHandlerTest.cpp
    ThreadPoolTest.cpp
    ${INC_ALL}
    TestMain.cpp        
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ../include/imh
        ../src/imh
)

target_link_libraries(${PROJECT_NAME}
    imagehandler
    logging
    confighandler
    gtest
    ${GMOCK_LIB}
)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
