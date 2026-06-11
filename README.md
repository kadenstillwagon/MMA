# Maximum Matching Accuracy: An Instance Segmentation Evaluation Metric Utilizing Globally Optimal Matching

[![Python 3.8+](https://img.shields.io/badge/python-3+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red.svg)](https://arxiv.org/abs/2606.10107)

Maximum Matching Accuracy (MMA) is a continuous, threshold-free metric for instance segmentation evaluation. It utilizes maximum bipartite matching to find a globally optimal one-to-one assignment between Ground Truth and Predicted masks and uses pixel-level normalization. 

## Quick Start

### Installation

```bash
# Install from source
git clone https://github.com/kadenstillwagon/MMA.git
cd MMA
pip install -e .
```

### Basic Usage

```python
import numpy as np
from mma.metrics.mma import get_mma

#Load Ground Truth and Predicted
ann = np.load('../mma/datasets/demo_example/ann.npy')
pred = np.load('../mma/datasets/demo_example/output.npy')

#Calculate
mma = get_mma(pred, ann, greedy=False, return_accuracy_plot=False)
print(mma)
```


## Citation

If you use MMA in your research, please cite our paper:

Add MMA Citation
```

## Support

- **Issues**: [GitHub Issues](https://github.com/kadenstillwagon/MMA/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kadenstillwagon/MMA/discussions)
- **Email**: kstillwagon26@gatech.edu

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Institutions

This work was developed at:
- **Georgia Institute of Technology**
  - School of Biological Sciences
  - School of Computer Science  
  - Department of Biomedical Engineering
  - School of Mechanical Engineering
  - PACE Computing Infrastructure

## Acknowledgments

- The open-source community for tools and datasets
- Georgia Tech for computational resources
- All contributors and users of DINOCell

---

