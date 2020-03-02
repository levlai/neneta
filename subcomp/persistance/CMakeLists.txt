cmake_minimum_required(VERSION 2.8)
project(persistance)

aux_source_directory(src/pers SRC_LIST)

file(GLOB_RECURSE INC_ALL "include/pers/*.h")

add_library(${PROJECT_NAME}
    STATIC
        ${SRC_LIST}
        ${INC_ALL}
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        include/pers
    PRIVATE
        src/pers
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        confighandler
        kompex-sqlite-wrapper        
)

