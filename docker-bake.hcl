variable "REPO_OWNER" {}
variable "VARIANT" {}
variable "SHA_SHORT" {}
variable "BUILD_NUMBER" {}
variable "LLAMA_COMMIT" {}
variable "CUDA_VERSION" {}
variable "CUDA_DOCKER_ARCH" { default = "86;90" }
variable "USE_CCACHE" { default = "true" }

# Common cache configuration for GitHub Actions
target "cache_settings" {
  cache-from = ["type=gha,scope=ccache-${VARIANT}"]
  cache-to   = ["type=gha,mode=max,scope=ccache-${VARIANT}"]
}

group "default" {
  targets = ["full", "swap"]
}

target "settings" {
  context = "."
  inherits = ["cache_settings"]
  args = {
    CUDA_VERSION     = "${CUDA_VERSION}"
    CUDA_DOCKER_ARCH = "${CUDA_DOCKER_ARCH}"
    USE_CCACHE       = "${USE_CCACHE}"
  }
}

target "full" {
  inherits = ["settings"]
  target = "full"
  tags = [
    "ghcr.io/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-full-${BUILD_NUMBER}-${LLAMA_COMMIT}",
    "ghcr.io/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-full",
    "ghcr.io/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-full"
  ]
}

target "swap" {
  inherits = ["settings"]
  target = "swap"
  tags = [
    "ghcr.io/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-swap-${BUILD_NUMBER}-${LLAMA_COMMIT}",
    "ghcr.io/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-swap",
    "ghcr.io/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-swap"
  ]
}
