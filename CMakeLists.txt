project(neneta)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
#set(CMAKE_CXX_FLAGS "-Wl,--stack,160000000 -std=c++11 -g -o0")
set(CMAKE_CXX_FLAGS "-std=c++11 -g -o0")
set(CMAKE_BUILD_TYPE "DEBUG")
add_definitions(-DARMA_32BIT_WORD=1)
ADD_DEFINITIONS(-DBOOST_LOG_DYN_LINK)

find_package(OpenCL REQUIRED)

#Set deployment directory
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/deploy)

#Shared library paths
if(WIN32)
    set(EXTERNAL_LIB_PATH "${CMAKE_SOURCE_DIR}/externals/libs/windows")
    set(XLIB "")
    file(GLOB_RECURSE DLLs ${EXTERNAL_LIB_PATH}/*.dll)
    find_library(GTEST_LIB gtest PATHS ${CMAKE_SOURCE_DIR}/externals/libs/windows)
    find_library(GMOCK_LIB gmock PATHS ${CMAKE_SOURCE_DIR}/externals/libs/windows)
else()
    set(EXTERNAL_LIB_PATH "${CMAKE_SOURCE_DIR}/externals/libs/linux/x86_64")
    set(XLIB X11)
    set(DLLIB "dl")
    file(GLOB_RECURSE DLLs ${EXTERNAL_LIB_PATH}/*)
    find_library(GTEST_LIB gtest PATHS ${CMAKE_SOURCE_DIR}/externals/libs/linux/x86_64)
    find_library(GMOCK_LIB gmock PATHS ${CMAKE_SOURCE_DIR}/externals/libs/linux/x86_64)
endif()
link_directories(${EXTERNAL_LIB_PATH})

#External includes
include_directories(${PROJECT_NAME}
    PRIVATE
        externals/include
        src/neneta
    SYSTEM
        ${OpenCL_INCLUDE_DIRS}
)

#Show xml conf files in project tree
file(GLOB_RECURSE CONF_FILES "configuration/*.xml")
file(GLOB_RECURSE INC_ALL "include/*.h")
file(GLOB_RECURSE TXT_FILES "*.txt")
file(GLOB_RECURSE RES_ALL "resources/*")


#Add subdirectories
add_subdirectory(subcomp)

add_executable(${PROJECT_NAME}
    src/neneta/main.cpp
    src/neneta/Neneta.cpp
    ${INC_ALL}
    ${CONF_FILES}
    ${TXT_FILES}
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        logging
        imagehandler
        imageprocessing
        confighandler
        gpgpu
        neuralnetwork
        ${OpenCL_LIBRARIES}
        ${XLIB}
        plotting
)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(FILES ${RES_ALL} DESTINATION .)
install(FILES ${CONF_FILES} DESTINATION .)
install(FILES ${DLLs} DESTINATION .)

