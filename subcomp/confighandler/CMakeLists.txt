cmake_minimum_required(VERSION 2.8)
project(confighandler)

aux_source_directory(src/conf SRC_LIST)

file(GLOB_RECURSE INC_ALL "include/conf/*.h")

add_library(${PROJECT_NAME}
    STATIC
        ${SRC_LIST}
        ${INC_ALL}
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        include/conf
    PRIVATE
        src/conf
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    gpgpu
    neuralnetwork
    common
)

