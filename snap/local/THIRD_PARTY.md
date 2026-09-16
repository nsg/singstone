# Third-party models

The Snap's model setup service downloads and verifies these checksum-pinned
files. The Snap itself contains their trusted manifest and notices:

- `ggml-base.en.bin`, from
  [whisper.cpp](https://github.com/ggml-org/whisper.cpp), MIT.
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
