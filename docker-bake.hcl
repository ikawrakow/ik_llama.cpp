# docker-bake.hcl

variable "SHA_SHORT" { default = "latest" }
variable "VARIANT" { default = "cpu" }
variable "CUDA_VERSION" { default = "none" }
variable "CUDA_DOCKER_ARCH" { default = "" }

group "default" {
  targets = ["full", "swap"]
}

target "base-settings" {
  context = "."
  dockerfile = "./docker/ik_llama-cuda.Containerfile" # or cpu, they share logic
  args = {
    CUDA_VERSION = "${CUDA_VERSION}"
    CUDA_DOCKER_ARCH = "${CUDA_DOCKER_ARCH}"
  }
  platforms = ["linux/amd64"]
}

target "full" {
  inherits = ["base-settings"]
  target = "full"
  tags = [
    "ghcr.io/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-full-${SHA_SHORT}",
    "ghcr.io/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-full"
  ]
}

target "swap" {
  inherits = ["base-settings"]
  target = "swap"
  tags = [
    "ghcr.io/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-swap-${SHA_SHORT}",
    "ghcr.io/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-swap"
  ]
}