# Local development override - automatically sets BUILD_NUMBER and BUILD_COMMIT
variable "BUILD_NUMBER" { default = "0" }
variable "BUILD_COMMIT" { default = "local-dev" }

target "server" {
  dockerfile = "./docker/ik_llama-cpu.Containerfile"
}

target "swap" {
  dockerfile = "./docker/ik_llama-cpu.Containerfile"
}

target "full" {
  dockerfile = "./docker/ik_llama-cpu.Containerfile"
}
