cmake_minimum_required(VERSION 2.8)
project(common)


file(GLOB_RECURSE INC_ALL "include/cmn/*.h")

add_library(${PROJECT_NAME}
    STATIC
        ${INC_ALL}
)

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        include/cmn
)
