# DMalNet
This is a malware detection and type classification framework using deep learning models. We exteact the API sequence and argument features, also generate the API call graphs and use GNNs to solve the malware detection and type classification problems. We provide the source code, model architecture, and API sequence data of demo test samples (including API call graphs generated by benign samples and malicious samples). If you want to reimplement this model against your own dataset, you need to extract the API sequence from the software sandbox report and process it into the form of test samples.

## Required Packages

- numpy
- scikit-learn
- gensim  
- dirty_cat
- pytorch (GPU)
- torch_geometric

## File Description
- `cuckoo_report`: the sandbox reports generated by Cuckoo Sandbox.
- `data`: the dataset demo after data preprocessing.
    - `sequence`: the feature vectors of each sample extracted by hybrid feature encoder.
    - `graph`: api call graph of each sample.
    - `data_test_demo.zip`: demo for model test.
        - `demo_classification_test.pt`: malware classification demo dataset.
        - `demo_detection_test.pt`: malware detection demo dataset.
- `encoder`: some trained encoders.
    - `skip-gram_previous.model`: skip_gram model.
    - `paths_sim.pkl`: similarity encoder for file paths.
    - `dlls_sim.pkl`: similarity encoder for DLLs.
    - `registry_sim.pkl`: similarity encoder for registry keys.
    - `urls_sim.pkl`: similarity encoder for URLs.
    - `ips_sim.pkl`: similarity encoder for IP address.
- `model`: trained models for testing (only GPU models).
    - `detection_model.pkl`: a binary classification model for malware detection trained on the whole training set (
      including malware and goodware).
    - `classification_model.pkl`: an 8-class classification model for malware classification trained on the malware training set.
- `Encoder_training.py`: the training process of word2vec and similarity encoders.
- `CuckooReport2FeatureVectors.py`: extracting feature vectors from cuckoo report of each sample based on the hybrid feature encoder.
- `graph_embedding.py`: API call graph generation.
- `data_split.py`: dataset split for training and test.
- `model_detection.py`: the graph learning model of malware detection, its detailed architecture, training process, and ablation study.
- `model_classification.py`: the graph learning model of malware classification, its detailed architecture, training process, and ablation study.  
- `experimental_figure.py`: ROC curve and confusion matrix generation.   
- `detection_model_test_demo.py`: test on the demo set of malware detection.
- `classification_model_test_demo.py`: test on the demo set of malware classification.

## Train & Test

Since the size of original data is too large(the original software samples are about 2TB, cuckoo reports about 3.4TB, feature vectors about 20GB), we just submit a test demo set to verify our models. Users can extract API call graph from their own malware samples using our code.

The malware detection demo set (i.e., demo_detection_test.pt) contains graphs generated by 1000 malwre and 1000 goodware.
The malware classification demo set (i.e., demo_classification_test.pt) contains graphs generated by malware of each type, and each type contains 125 samples. Each graph contains 4 attributes:
```
data_test = torch.load('./data/demo_detection_test.pt')
graph = data_test[0]
garph.x             # node feature vector of each node
graph.edge_index    # edges between nodes <v_i, v_j>
graph.edge_attr     # edge feature vector of each edge
garph.y             # label of the graph
```

The complete code execution sequence is as follows:
- `Encoder_training.py`
- `CuckooReport2FeatureVectors.py`
- `graph_embedding.py`
- `dataset_split.py`
- `model_detection.py` & `model_classification.py`

The model test code execution sequence is as follows:
- extract the `data_test_demo.zip` and get `demo_classification_test.pt` and `demo_detection_test.pt`.
- run `detection_model_test_demo.py` or `classification_model_test_demo.py` on GPU.

## DMalNet

```
@article{li2022dmalnet,
  title={DMalNet: Dynamic Malware Analysis Based on API Feature Engineering and Graph Learning},
  author={Li, Ce and Cheng, Zijun and Zhu, He and Wang, Leiqi and Lv, Qiujian and Wang, Yan and Li, Ning and Sun, Degang},
  journal={Computers \& Security},
  pages={102872},
  year={2022},
  publisher={Elsevier}
}
```
