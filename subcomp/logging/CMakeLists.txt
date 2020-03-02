cmake_minimum_required(VERSION 2.8)
project(logging)

include_directories(${PROJECT_NAME}
    PRIVATE
        include/log
)

aux_source_directory(src/log SRC_LIST)

file(GLOB_RECURSE INC_ALL "include/*.h")

add_library(${PROJECT_NAME}
    STATIC
        ${SRC_LIST}
        ${INC_ALL}
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        include/log
        src/log
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        confighandler
        boost_log
        boost_log_setup
        boost_system
        boost_regex
        boost_filesystem
        boost_date_time
        boost_atomic
        boost_thread
        pthread
)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
