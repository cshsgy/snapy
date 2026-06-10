include(FetchContent)
set(FETCHCONTENT_QUIET TRUE)

# commux: UCX tag-matching c10d backend (https://github.com/zoeyzyhu/commux).
# It owns UCX discovery/bundling -- if no system UCX is found it builds UCX
# (+gdrcopy) from source -- so snapy needs no UCX setup of its own. Point it at a
# prebuilt UCX with -DUCX_ROOT=/path to skip the source build.
set(PACKAGE_NAME commux)
set(REPO_URL "https://github.com/zoeyzyhu/commux")
set(REPO_TAG "v0.1.0")

add_package(${PACKAGE_NAME} ${REPO_URL} ${REPO_TAG} "" ON)

set(COMMUX_LIBRARY commux::commux CACHE STRING "commux library name")
