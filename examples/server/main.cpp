// llama-server executable entry point
// the actual server logic lives in server.cpp (llama_server), which is also
// used by the CLI (llama-cli) via --server-base

int llama_server(int argc, char ** argv);

int main(int argc, char ** argv) {
    return llama_server(argc, argv);
}
