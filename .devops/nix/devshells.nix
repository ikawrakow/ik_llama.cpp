{ inputs, ... }:

{
  perSystem =
    {
      config,
      lib,
      system,
      ...
    }:
    {
      devShells =
        let
          pkgs = inputs.nixpkgs.legacyPackages.${system};
          inherit (pkgs) stdenv;
        in
        lib.concatMapAttrs (
          name: package: {
            ${name} = pkgs.mkShell {
              name = "${name}";
              inputsFrom = [ package ];
              shellHook = ''
                echo "Entering ${name} devShell"
              '';
            };
            "${name}-extra" = pkgs.mkShell {
              name = "${name}-extra";
              inputsFrom = [ package ];
              packages = with pkgs.python3Packages; [
                numpy
                sentencepiece
                tiktoken
                torchWithoutCuda
                transformers
              ];
              shellHook = ''
                echo "Entering ${name}-extra devShell"
                addToSearchPath "LD_LIBRARY_PATH" "${lib.getLib stdenv.cc.cc}/lib"
              '';
            };
          }
        ) config.packages;
    };
}
