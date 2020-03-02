cmake_minimum_required(VERSION 2.8)
project(gpgpu)

aux_source_directory(src/gpu SRC_LIST)

file(GLOB_RECURSE INC_ALL "include/gpu/*.h")

add_library(${PROJECT_NAME}
    STATIC
        ${SRC_LIST}
        ${INC_ALL}
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        include/gpu
    PRIVATE
        src/gpu
)

target_link_libraries(${PROJECT_NAME}
    confighandler
    logging
    ${OpenCL_LIBRARIES}
    common
)
