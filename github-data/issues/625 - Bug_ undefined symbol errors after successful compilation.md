### 🐛 [#625](https://github.com/ikawrakow/ik_llama.cpp/issues/625) - Bug: undefined symbol errors after successful compilation

| **Author** | `samteezy` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-17 |
| **Updated** | 2025-07-18 |

---

#### Description

### What happened?

I'm a bit of a newbie here, and apologies if I'm doing something wrong.

I'm currently compiling llama.cpp and running with llama-swap, and all is well. I decided to give this fork a try alongside my current setup.

I can compile ik_llama, but when I go to run llama-cli or llama-server (even to just get the current version), I get this error: 

`/root/llama-builds/ik_llama.cpp/bin/llama-server: undefined symbol: llama_set_offload_policy`

Build flags:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DGGML_CCACHE=OFF \
    -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=FLAME \
    -DGGML_VULKAN=ON
```

These are similar to my llama.cpp build, but that uses HIP/ROCm instead of Vulkan. (note I have tried this both with Vulkan ON and OFF with same result).

I do see these warnings in the build logs:

```bash
[  9%] Built target build_info
[ 10%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_256_256.cpp.o
In function 'SHA1Update',
    inlined from 'SHA1Final' at /root/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:265:5:
/root/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:219:13: warning: 'SHA1Transform' reading 64 bytes from a region of size 0 [-Wstringop-overread]
  219 |             SHA1Transform(context->state, &data[i]);
      |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/root/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:219:13: note: referencing argument 2 of type 'const unsigned char[64]'
/root/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c: In function 'SHA1Final':
/root/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:54:6: note: in a call to function 'SHA1Transform'
   54 | void SHA1Transform(
      |      ^~~~~~~~~~~~~
In function 'SHA1Update',
    inlined from 'SHA1Final' at /root/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:269:9:
/root/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:219:13: warning: 'SHA1Transform' reading 64 bytes from a region of size 0 [-Wstringop-overread]
  219 |             SHA1Transform(context->state, &data[i]);
      |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/root/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:219:13: note: referencing argument 2 of type 'const unsigned char[64]'
/root/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c: In function 'SHA1Final':
/root/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:54:6: note: in a call to function 'SHA1Transform'
   54 | void SHA1Transform(
      |      ^~~~~~~~~~~~~
[ 10%] Built target sha256
[ 10%] Built target sha1
[ 10%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_128_128.cpp.o
```

...but that's all that stands out to me.

### Name and Version

Current main branch

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
Ubuntu 24.04 running in Proxmox LXC
```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-07-18** at **06:12:11**:<br>

It looks like a confusion between `llama.cpp` and `ik_llama.cpp` libraries. I suspect `llama.cpp` is installed system-wide, so when the `ik_llama.cpp` server is started it picks up the `llama.cpp` DLLs. 

This project does not consider the possibility of co-existing with a system-wide installation of `llama.cpp`. The work around is to use `LD_LIBRARY_PATH`, e.g.,
```
export LD_LIBRARY_PATH="/root/llama-builds/ik_llama.cpp/bin:$LD_LIBRARY_PATH"
/root/llama-builds/ik_llama.cpp/bin/llama-server ...
```

---

👤 **samteezy** commented the **2025-07-18** at **12:14:29**:<br>

Yep, that was root cause. I've been restructuring my llama environment to use local, static builds of both `llama.cpp` and `ik_llama.cpp` this morning using `-DBUILD_SHARED_LIBS=OFF` and now they're both working great.
Thanks for all your hard work!