cmake_minimum_required(VERSION 2.8)
project(imagehandler)

aux_source_directory(src/imh SRC_LIST)

file(GLOB_RECURSE INC_ALL "include/imh/*.h" "src/imh/*.h")

#add_subdirectory(test)

add_library(${PROJECT_NAME}
    STATIC
        ${SRC_LIST}
        ${INC_ALL}
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        include/imh
    PRIVATE
        src/imh
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        kompex-sqlite-wrapper
        confighandler
        boost_filesystem
        imageprocessing
        ${DLLIB}
)

