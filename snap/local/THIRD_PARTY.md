# Third-party models

The Snap's model setup service downloads and verifies these checksum-pinned
files. The Snap itself contains their trusted manifest and notices:

- `kb-whisper-small-q5_0.bin`, the standard Swedish KB-Whisper Small model
  from [KBLab](https://huggingface.co/KBLab/kb-whisper-small), Apache 2.0. Its
  license is installed as `licenses/kb-whisper-small.LICENSE`.
- `segmentation.onnx`, the
  [pyannote segmentation 3.0](https://huggingface.co/pyannote/segmentation-3.0)
  model converted to ONNX by
  [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), MIT. Its license text
  is installed as `licenses/pyannote-segmentation.LICENSE`.
- `nemo_en_titanet_small.onnx`, derived from NVIDIA NeMo TitaNet-S and
  distributed by
  [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), CC BY 4.0.

Exact revisions, source URLs, sizes, and SHA-256 digests are recorded in the
installed `models.lock` file.

## Bundled GPU runtimes

- Intel oneAPI DPC++/C++ and Unified Runtime 2026.1 provide the SYCL runtime.
- Intel oneMKL 2026.1 provides the SYCL BLAS implementation used by ggml.
- The Level Zero loader and tracing/validation layers 1.34.0 are built from a
  pinned upstream source commit; their MIT license is installed as
  `level-zero-LICENSE`.
- The OpenCL loader and Intel compute runtime come from Ubuntu Noble's
  `libze-intel-gpu1` and `intel-opencl-icd` packages. Their package copyright
  and license files are included under `/usr/share/doc`.

The Intel component licenses and third-party notices copied from the build
packages are installed beside this file under `oneapi-licensing` and
`onemkl-licensing`. Ubuntu package copyright files are installed under
`/usr/share/doc` in the Snap.
