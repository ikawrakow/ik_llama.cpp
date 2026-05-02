variable "REPO_OWNER" { default = "local" }
variable "VARIANT" { default = "cpu" }
variable "BUILD_NUMBER" { default = "0" }
variable "CUDA_VERSION" {}
variable "CUDA_DOCKER_ARCH" { default = "86;90" }
variable "USE_CCACHE" { default = "true" }
variable "GGML_NATIVE" { default = "ON" }

# Common cache configuration for GitHub Actions
target "cache_settings" {
  cache-from = ["type=gha,scope=ccache-${VARIANT}"]
  cache-to   = ["type=gha,mode=max,scope=ccache-${VARIANT}"]
}

group "default" {
  targets = ["server", "full", "swap"]
}

target "settings" {
  context = "."
  inherits = ["cache_settings"]
  args = {
    BUILD_NUMBER     = "${BUILD_NUMBER}"
    CUDA_VERSION     = "${CUDA_VERSION}"
    CUDA_DOCKER_ARCH = "${CUDA_DOCKER_ARCH}"
    GGML_NATIVE      = "${GGML_NATIVE}"
    USE_CCACHE       = "${USE_CCACHE}"
  }
}

target "server" {
  inherits = ["settings"]
  target = "server"
  tags = [
    "ghcr.io/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-server-${BUILD_NUMBER}",
    "ghcr.io/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-server"
  ]
}

target "full" {
  inherits = ["settings"]
  target = "full"
  tags = [
    "ghcr.io/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-full-${BUILD_NUMBER}",
    "ghcr.io/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-full"
  ]
}

target "swap" {
  inherits = ["settings"]
  target = "swap"
  tags = [
    "ghcr.io/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-swap-${BUILD_NUMBER}",
    "ghcr.io/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-swap"
  ]
}