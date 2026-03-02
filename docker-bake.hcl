variable "REPO_OWNER" {}
variable "VARIANT" {}
variable "SHA_SHORT" {}
variable "CUDA_VERSION" {}
variable "CUDA_DOCKER_ARCH" { default = "86;90" }

group "default" {
  targets = ["full", "swap"]
}

target "settings" {
  context = "."
  args = {
    CUDA_VERSION = "${CUDA_VERSION}"
    CUDA_DOCKER_ARCH = "${CUDA_DOCKER_ARCH}"
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