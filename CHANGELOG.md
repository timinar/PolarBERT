# Changelog

## [v0.2] - 2025-04-02

### Added

* Support for the (unreleased) Prometheus dataset.
* Support for hyperparameter sweeps.
* Add neutrino energy regression as a downstream task.
* Allow training from scratch in `finetuning.py` by passing `new` as the checkpoint path.
* Optionally add a random offset to the timestamps of each event, as a data augmentation.

### Changed

* Assign equal weights to all events in the loss, instead of assigning equal weights to all pulses as done previously.
* Slightly modify the format of YAML config files, and adjust the config templates accordingly.

### Fixed

* Use the correct dataset length when the number of requested events exceeds the dataset size.
* Correctly load PyTorch-Lightning checkpoints (.ckpt) in addition to plain PyTorch checkpoints (.pth).

## [v0.1] - 2025-01-24

### Added

* Initial public release of PolarBERT, matching [our contribution](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_259.pdf) to the NeurIPS 2024 workshop "Machine Learning and the Physical Sciences".
