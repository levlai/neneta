cmake_minimum_required(VERSION 2.8)
project(imageprocessing)

include_directories(${PROJECT_NAME}
    PRIVATE
        include/ip
)

aux_source_directory(src/ip SRC_LIST)

file(GLOB_RECURSE INC_ALL "include/*.h")

add_library(${PROJECT_NAME}
    STATIC
        ${SRC_LIST}
        ${INC_ALL}
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        include/ip
        src/ip
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        jpeg
        logging
        common
)

