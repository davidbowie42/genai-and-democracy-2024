let
pkgs = import <nixpkgs> {};
in pkgs.mkShell {
	packages = [
					(pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
					   pandas
					   requests
					   torch
					   numpy
					   transformers
					   datasets
					   evaluate
					   accelerate
					   huggingface-hub
					   ollama
					   jupyter
					   pip
					   pydantic
					   openai
					   instructor
					   typing
		]))

	];
}
