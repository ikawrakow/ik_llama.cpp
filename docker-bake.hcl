variable "REPO_OWNER" {}
variable "VARIANT" {}
variable "SHA_SHORT" {}
variable "CUDA_VERSION" {}
variable "CUDA_DOCKER_ARCH" { default = "86;90" }

# Define common cache settings to avoid repetition
target "cache_settings" {
  cache-from = ["type=gha,scope=ccache-${VARIANT}"]
  cache-to   = ["type=gha,mode=max,scope=ccache-${VARIANT}"]
}

group "default" {
  targets = ["full", "swap"]
}

target "settings" {
  context = "."
  # Inherit from cache_settings so all targets use the GHA cache
  inherits = ["cache_settings"]
  args = {
    CUDA_VERSION = "${CUDA_VERSION}"
    CUDA_DOCKER_ARCH = "${CUDA_DOCKER_ARCH}"
    # Pass ccache dir as an arg if your Dockerfile needs to know the path
    CCACHE_DIR = "/ccache" 
  }
}

target "full" {
  inherits = ["settings"]
  target = "full"
  tags = [
    "ghcr.io/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-full-${SHA_SHORT}",
    "ghcr.io/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-full"
  ]
}

target "swap" {
  inherits = ["settings"]
  target = "swap"
  tags = [
    "ghcr.io/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-swap-${SHA_SHORT}",
    "ghcr.io/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-swap"
  ]
}
