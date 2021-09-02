check_and_exit() {
  if [[ $? -ne 0 ]]; then
    echo "Failed"
    exit 1
  fi
}

download_pkg() {
  pkg_name=$1
  prefix_name=$2
  pkg_url=$3
  mkdir -p "${pkg_name}-download/${prefix_name}-prefix/src/"
  check_and_exit
  cd "${pkg_name}-download/${prefix_name}-prefix/src/"
  check_and_exit
  wget ${pkg_url}
  check_and_exit
  cd -
}

mode=$1
if [[ "X${mode}" == "X?" ]];then
  echo "./example full/cmake/make/deploy, default deploy"
  exit 0
fi
if [[ "X${mode}" == "X" ]];then
  mode=deploy
fi
mkdir -p build
check_and_exit
cd build
check_and_exit
if [[ "X${mode}" == "Xfull" ]];then
  rm -rf *
  download_pkg clog clog https://github.com/pytorch/cpuinfo/archive/d5e37adf1406cf899d7d9ec1d317c47506ccb970.tar.gz
  download_pkg cpuinfo cpuinfo https://github.com/pytorch/cpuinfo/archive/5916273f79a21551890fd3d56fc5375a78d1598d.zip
  download_pkg FP16 fp16 https://github.com/Maratyszcza/FP16/archive/0a92994d729ff76a58f692d3028ca1b64b145d91.zip
  download_pkg FXdiv fxdiv https://github.com/Maratyszcza/FXdiv/archive/b408327ac2a15ec3e43352421954f5b1967701d1.zip
  download_pkg googlebenchmark googlebenchmark https://github.com/google/benchmark/archive/v1.5.3.zip
  download_pkg googletest googletest https://github.com/google/googletest/archive/5a509dbd2e5a6c694116e329c5a20dc190653724.zip
  download_pkg pthreadpool pthreadpool https://github.com/Maratyszcza/pthreadpool/archive/545ebe9f225aec6dca49109516fac02e973a3de2.zip
fi
if [[ "X${mode}" == "Xfull" || "X${mode}" == "Xcmake" ]];then
  cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL="19" \
    -DANDROID_NDK="${ANDROID_NDK}" -DANDROID_ABI="arm64-v8a" -DANDROID_TOOLCHAIN_NAME="aarch64-linux-android-clang"  \
    -DANDROID_STL=c++_shared -DXNNPACK_BUILD_TESTS=off -DXNNPACK_BUILD_BENCHMARKS=off ..
  check_and_exit
fi
if [[ "X${mode}" == "Xfull" || "X${mode}" == "Xcmake" || "X${mode}" == "Xmake" ]];then
  make -j10
  check_and_exit
fi
#sshpass -p root scp example/example root@10.175.96.127:/root/hgq/
#check_and_exit
#sshpass -p root ssh root@10.175.96.127 "adb -s JTK push /root/hgq/example/ /data/local/tmp/hgq/sparsity"
#check_and_exit
#sshpass -p root ssh root@10.175.96.127 "adb -s JTK shell 'LD_LIBRARY_PATH=/data/local/tmp/hgq/sparsity/ /data/local/tmp/hgq/sparsity/example'"
#check_and_exit

adb push example/example /data/local/tmp/hgq/sparsity
check_and_exit
adb shell "LD_LIBRARY_PATH=/data/local/tmp/hgq/sparsity/ /data/local/tmp/hgq/sparsity/example"
check_and_exit
echo "Done"
