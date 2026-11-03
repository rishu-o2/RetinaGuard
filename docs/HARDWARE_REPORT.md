# RetinaGuard Hardware & Environment Detection Report

Generated via automated empirical hardware detection.

---

## 1. GPU Detection

| Component | Measured Value | Notes / Error Output |
| :--- | :--- | :--- |
| **`nvidia-smi` Status** | **Missing** | `CommandNotFoundException: The term 'nvidia-smi' is not recognized as the name of a cmdlet, function, script file, or operable program.` |
| **GPU Hardware** | **Intel(R) Iris(R) Xe Graphics** | Integrated GPU (`AdapterRAM: 2,147,479,552 bytes` / 2.0 GB shared) |
| **Discrete NVIDIA GPU** | **None detected** | No NVIDIA hardware present on system |
| **Driver Version** | `32.0.101.7076` | Intel Graphics Driver |
| **CUDA Available (`torch.cuda.is_available()`)** | `False` | No CUDA-capable device |
| **PyTorch Build Type** | `2.14.0+cpu` | CPU-only build |
| **CUDA Install Command (for reference)** | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` | *Do not run on this machine; no NVIDIA GPU is present.* |

---

## 2. System Hardware & Storage Resources

| Resource | Measured Value |
| :--- | :--- |
| **CPU Model** | `13th Gen Intel(R) Core(TM) i5-1335U` |
| **Physical Cores** | `10` |
| **Logical Processors** | `12` |
| **Total System RAM** | `15.65 GB` |
| **D: Drive Total Space** | `166.02 GB` |
| **D: Drive Free Space** | `103.22 GB` |
| **D: Drive Used Space** | `62.80 GB` |

---

## 3. Software & Environment Versions

| Package / Environment | Version |
| :--- | :--- |
| **Operating System** | `Windows-11-10.0.26200-SP0` |
| **Python** | `3.14.3` |
| **torch** | `2.14.0+cpu` |
| **torchvision** | `0.29.0+cpu` |
| **timm** | `NOT INSTALLED` |
| **onnxruntime** | `1.30.0` |

---

## 4. Benchmark

*Skipped: No CUDA GPU available (`torch.cuda.is_available() == False`).*

