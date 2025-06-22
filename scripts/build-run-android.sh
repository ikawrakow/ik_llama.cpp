#!/bin/bash
#
# build ik_llama.cpp on Linux for Android phone
#
set -e

######## part-1 ########

PWD=`pwd`
PROJECT_HOME_PATH=`pwd`
PROJECT_ROOT_PATH=${PROJECT_HOME_PATH}
HOST_CPU_COUNTS=`cat /proc/cpuinfo | grep "processor" | wc | awk '{print int($1)}'`

#running path on Android phone
REMOTE_PATH=/data/local/tmp/

#Android NDK can be found at:
#https://developer.android.com/ndk/downloads
ANDROID_PLATFORM=android-34
ANDROID_NDK_VERSION=r28
ANDROID_NDK_NAME=android-ndk-${ANDROID_NDK_VERSION}
ANDROID_NDK_FULLNAME=${ANDROID_NDK_NAME}-linux.zip
ANDROID_NDK=${PROJECT_ROOT_PATH}/prebuilts/${ANDROID_NDK_NAME}

######## part-2 ########

#ref:https://github.com/ikawrakow/ik_llama.cpp/pull/429
#ENABLE_GGML_IQK_FLASH_ATTENTION=ON
ENABLE_GGML_IQK_FLASH_ATTENTION=OFF

RUNNING_PARAMS=" -ngl 99 -t 8 -n 256 "

PROMPT_STRING="introduce the movie Once Upon a Time in America briefly.\n"

#for llama-cli & llama-bench, 1.12 GiB, will be downloaded automatically via this script
TEST_MODEL_NAME=/sdcard/qwen1_5-1_8b-chat-q4_0.gguf

######## part-3: utilities and functions ########

function dump_vars()
{
    echo -e "ANDROID_NDK:          ${ANDROID_NDK}"
}


function show_pwd()
{
    echo -e "current working path:$(pwd)\n"
}


function check_command_in_host()
{
    set +e
    cmd=$1
    ls /usr/bin/${cmd}
    if [ $? -eq 0 ]; then
        #printf "${cmd} already exist on host machine\n"
        echo ""
    else
        printf "${cmd} not exist on host machine, pls install command line utility ${cmd} firstly and accordingly\n"
        exit 1
    fi
    set -e
}


function check_commands_in_host()
{
    check_command_in_host wget
}


function check_and_download_ndk()
{
    mkdir -p ${PROJECT_ROOT_PATH}/prebuilts

    is_android_ndk_exist=1

    if [ ! -d ${ANDROID_NDK} ]; then
        is_android_ndk_exist=0
    fi

    if [ ! -f ${ANDROID_NDK}/build/cmake/android.toolchain.cmake ]; then
        is_android_ndk_exist=0
    fi

    if [ ${is_android_ndk_exist} -eq 0 ]; then

        if [ ! -f ${PROJECT_ROOT_PATH}/prebuilts/${ANDROID_NDK_FULLNAME} ]; then
            wget --no-config --quiet --show-progress -O ${PROJECT_ROOT_PATH}/prebuilts/${ANDROID_NDK_FULLNAME} https://dl.google.com/android/repository/${ANDROID_NDK_FULLNAME}
        fi

        cd ${PROJECT_ROOT_PATH}/prebuilts/
        unzip ${ANDROID_NDK_FULLNAME}

        if [ $? -ne 0 ]; then
            printf "failed to download Android NDK to %s \n" "${ANDROID_NDK}"
            exit 1
        fi
        cd ${PROJECT_ROOT_PATH}

        printf "Android NDK saved to ${ANDROID_NDK} \n\n"
    else
        printf "Android NDK already exist:         ${ANDROID_NDK} \n\n"
    fi
}


function build_arm64
{
    cmake -H. -B./out/android -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DLLAMA_CURL=OFF -DGGML_IQK_FLASH_ATTENTION=${ENABLE_GGML_IQK_FLASH_ATTENTION}
    cd out/android
    make -j${HOST_CPU_COUNTS}
    show_pwd

    cd -
}


function build_arm64_debug
{
    cmake -H. -B./out/android -DCMAKE_BUILD_TYPE=Debug -DGGML_OPENMP=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DLLAMA_CURL=OFF -DGGML_IQK_FLASH_ATTENTION=${ENABLE_GGML_IQK_FLASH_ATTENTION}
    cd out/android
    make -j${HOST_CPU_COUNTS}
    show_pwd

    cd -
}


function remove_temp_dir()
{
    if [ -d out/android ]; then
        echo "remove out/android directory in `pwd`"
        rm -rf out/android
    fi
}


function build_ik_llamacpp()
{
    show_pwd
    check_and_download_ndk
    dump_vars
    remove_temp_dir
    build_arm64
}


function build_ik_llamacpp_debug()
{
    show_pwd
    check_and_download_ndk
    dump_vars
    remove_temp_dir
    build_arm64_debug
}


function check_and_download_model()
{
    set +e

    model_name=$1
    model_url=$2

    adb shell ls /sdcard/${model_name}
    if [ $? -eq 0 ]; then
        printf "the prebuild LLM model ${model_name} already exist on Android phone\n"
    else
        printf "the prebuild LLM model ${model_name} not exist on Android phone\n"
        wget --no-config --quiet --show-progress -O ${PROJECT_ROOT_PATH}/models/${model_name} ${model_url}
        adb push ${PROJECT_ROOT_PATH}/models/${model_name} /sdcard/
    fi

    set -e
}


function check_prebuilt_models()
{
    #https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/blob/main/gemma-3-4b-it-Q8_0.gguf,              size 4.13 GiB
    #https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-GGUF/blob/main/qwen1_5-1_8b-chat-q4_0.gguf,          size 1.12 GiB

    set +e

    check_and_download_model qwen1_5-1_8b-chat-q4_0.gguf https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-GGUF/resolve/main/qwen1_5-1_8b-chat-q4_0.gguf
    set -e
}


function prepare_run_on_phone()
{
    if [ $# != 1 ]; then
        print "invalid param"
        return
    fi
    program=$1

    check_prebuilt_models

    all_libs=`find ./out/android -name "*.so" -print`
    #echo ${all_libs}
    for lib in ${all_libs};do
        adb push ${lib} ${REMOTE_PATH}/
    done

    adb push ./out/android/bin/${program} ${REMOTE_PATH}/

    adb shell chmod +x ${REMOTE_PATH}/${program}
}


function run_llamacli()
{
    prepare_run_on_phone llama-cli

    echo "${REMOTE_PATH}/llama-cli ${RUNNING_PARAMS} -m ${TEST_MODEL_NAME} -p \"${PROMPT_STRING}\""
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-cli ${RUNNING_PARAMS} -m ${TEST_MODEL_NAME} -p \"${PROMPT_STRING}\""

}


function run_llamabench()
{
    prepare_run_on_phone llama-bench

    echo "adb shell \"cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-bench ${RUNNING_PARAMS} -m ${TEST_MODEL_NAME}\""
    echo "${REMOTE_PATH}/llama-bench ${RUNNING_PARAMS} -m ${TEST_MODEL_NAME}"

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-bench ${RUNNING_PARAMS} -m ${TEST_MODEL_NAME}"

}


function show_usage()
{
    echo -e "\n\n\n"
    echo "Usage:"
    echo "  $0 help"
    echo "  $0 build"
    echo "  $0 build_debug"
    echo "  $0 run_llamacli"
    echo "  $0 run_llamabench"

    echo -e "\n\n\n"
}


######## part-4: entry point  ########

show_pwd

check_commands_in_host
check_and_download_ndk
check_prebuilt_models

if [ $# == 0 ]; then
    show_usage
    exit 1
elif [ $# == 1 ]; then
    if [ "$1" == "-h" ]; then
        show_usage
        exit 1
    elif [ "$1" == "help" ]; then
        show_usage
        exit 1
    elif [ "$1" == "build" ]; then
        build_ik_llamacpp
        exit 0
    elif [ "$1" == "build_debug" ]; then
        build_ik_llamacpp_debug
        exit 0
    elif [ "$1" == "run_llamacli" ]; then
        run_llamacli
        exit 0
    elif [ "$1" == "run_llamabench" ]; then
        run_llamabench
        exit 0
    else
        show_usage
        exit 1
    fi
else
    show_usage
    exit 1
fi
