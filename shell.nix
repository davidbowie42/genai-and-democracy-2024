{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell
{
  nativeBuildInputs = with pkgs; [
    python3  
    python312Packages.transformers
    python312Packages.torch
    python312Packages.numpy
  ];

  shellHook = ''
    source ./venv/bin/activate
  '';

  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib/:/run/opengl-driver/lib/";
}
