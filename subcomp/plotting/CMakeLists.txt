cmake_minimum_required(VERSION 2.8)
project(plotting)

aux_source_directory(src/plot SRC_LIST)

file(GLOB_RECURSE INC_ALL "include/plot/*.h")

add_library(${PROJECT_NAME}
    STATIC
        ${SRC_LIST}
        ${INC_ALL}
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        include/plot
    PRIVATE
        src/plot
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        confighandler
        boost_iostreams
        boost_system
        boost_filesystem 
)

