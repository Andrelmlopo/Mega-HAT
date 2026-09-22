# Third-party components

Mega-HAT's original code is MIT licensed. The adapted coarse cache uses Apache
2.0 and the adapted DROID readout uses BSD 3-Clause, with notices retained.

Mega-HAT contains the temporal method, sequence interface, object preparation
and model adapters. It does not bundle upstream neural model sources,
checkpoints, CAD models or datasets.

- **MegaPose**, Yann Labbé, Lucas Manuelli, Arsalan Mousavian, Stephen Tyree,
  Stan Birchfield, Jonathan Tremblay, Justin Carpentier, Mathieu Aubry,
  Dieter Fox and Josef Sivic. [Repository](https://github.com/megapose6d/megapose6d).
  Apache License 2.0. Its notice is retained in
  [licenses/MegaPose.txt](licenses/MegaPose.txt). The compatibility patch modifies
  the specified upstream files. The coarse-cache adapter follows its inference
  procedure and retains the same Apache terms. Model weights retain the
  upstream model license.
- **DROID-SLAM**, Zachary Teed and Jia Deng.
  [Repository](https://github.com/princeton-vl/DROID-SLAM). BSD 3-Clause.
  The causal readout adapts its trajectory-filling procedure. Its notice is in
  [licenses/DROID-SLAM.txt](licenses/DROID-SLAM.txt).
- **Panda3D**, **panda3d-gltf**, **trimesh**, **NumPy**, **SciPy**, **Pillow**,
  **PyTorch**, **OpenCV**, **Pinocchio**, and other separately installed
  dependencies retain their respective licenses.

```bibtex
@inproceedings{labbe2022megapose,
  title={MegaPose: 6D Pose Estimation of Novel Objects via Render \& Compare},
  author={Labb{\'e}, Yann and Manuelli, Lucas and Mousavian, Arsalan and
          Tyree, Stephen and Birchfield, Stan and Tremblay, Jonathan and
          Carpentier, Justin and Aubry, Mathieu and Fox, Dieter and Sivic, Josef},
  booktitle={Conference on Robot Learning},
  year={2022}
}

@article{teed2021droid,
  title={DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras},
  author={Teed, Zachary and Deng, Jia},
  journal={Advances in Neural Information Processing Systems},
  year={2021}
}
```
