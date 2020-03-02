cmake_minimum_required(VERSION 2.8)
project(neuralnetwork)

include_directories(${PROJECT_NAME}
    PRIVATE
        include/net
)

aux_source_directory(src/net SRC_LIST)
file(GLOB_RECURSE CL_FILES "src/net/*.cl*")

file(GLOB_RECURSE INC_ALL "include/*.h")

add_subdirectory(test)
add_library(${PROJECT_NAME}
    STATIC
        ${SRC_LIST}
        ${INC_ALL}
        ${CL_FILES}
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        include/net
        src/net
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        gpgpu
        imageprocessing
        persistance
        logging
        common
)

install(FILES ${CL_FILES} DESTINATION .)
