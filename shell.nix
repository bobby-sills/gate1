# Dev shell for the gate experiments. `nix-shell` in this directory, then:
#
#     python selftest.py      # Gate 1 pipeline, fake backend
#     python selftest2.py     # Gate 2 pipeline, fake backend
#     python selftest2b.py    # Gate 2b pipeline, fake backend
#
# These are the CPU-side dependencies only. torch and transformers are deliberately absent:
# nothing that needs them runs here -- the real backend runs on Colab (see COLAB.md), and
# every selftest substitutes a fake backend precisely so the pipeline can be exercised
# without a GPU or a 16GB model download.
{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (ps: with ps; [
      numpy
      pandas
      scikit-learn
      scipy
    ]))
  ];
}
