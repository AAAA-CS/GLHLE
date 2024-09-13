## Learning Hyperspectral Noisy Label with Global and Local Hypergraph Laplacian Energy <br>


> Abstract: Deep learning has achieved significant advancements in hyperspectral image (HSI) classification, yet it is highly dependent on the availability of high-quality labeled data. However, acquiring such labeled data for HSIs is often challenging due to the associated high costs and complexity. Consequently, the issue of classifying HSIs with noisy labels has garnered increasing attention. To address the negative effects of noisy labels, various methods have employed label correction strategies and have demonstrated promising results. Nevertheless, these techniques typically rely on correcting labels based on small-loss samples or neighborhood similarity. In high-noise environments, such methods often face unstable training processes, and the unreliability of neighborhood samples restricts their effectiveness. To overcome these limitations, this paper proposes a label correction method designed to address noisy labels in HSI classification by leveraging both global and local hypergraph structures to estimate label confidence and correct mislabeled samples. In contrast to traditional graph-based approaches, hypergraphs are capable of capturing higher-order relationships among samples, thereby improving the accuracy of label correction. The proposed method minimizes both global and local hypergraph Laplacian energies to enhance label consistency and accuracy across the dataset. Furthermore, contrastive learning and the Mixup technique are integrated to bolster the robustness and discriminative capabilities of HSI classification networks. Extensive experiments conducted on four publicly available hyperspectral datasets—University of Pavia (UP), Salinas Valley (SV), Kennedy Space Center (KSC), and WHU-Hi-HanChuan (HC)—demonstrate the superior performance of the proposed method, particularly in scenarios characterized by high levels of noise, where substantial improvements in classification accuracy are observed.


### Training

The config files are in `models/glhle/configs/`, just run the following command:
```asp
                        $ python main.py
```
