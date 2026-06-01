# Explainable_CV

# Results for ResNet50 ( Frozen )

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 92.64% |
| Precision | 92.64% |
| Recall    | 92.64% |
| F1 Score  | 92.61% |

## Confusion Matrix

![CM](notebooks/resnet50_confusion_matrix.png)

# Grad-CAM Explainability Analysis

## A. Correct COVID Predictions

| Sample         | Observation                                                                                                                                               |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| COVID Sample 1 | Grad-CAM strongly activates bilateral lower lung regions, indicating that the model is focusing on pulmonary infiltrates associated with COVID infection. |
| COVID Sample 2 | Concentrated activations are visible around patchy opacities in the right lung, suggesting meaningful pathological localization.                          |
| COVID Sample 3 | Heatmap highlights diffuse mid-lung abnormalities with minimal background activation, showing anatomically relevant attention.                            |
| COVID Sample 4 | Model attention is localized around peripheral lung regions, consistent with common COVID radiographic manifestations.                                    |
| COVID Sample 5 | Strong bilateral activation indicates the model successfully captured abnormal pulmonary textures instead of irrelevant structures.                       |

---

## B. Correct Normal Predictions

| Sample          | Observation                                                                                                              |
| --------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Normal Sample 1 | Activations are weak and diffuse with no concentrated pathological hotspots, indicating healthy lung interpretation.     |
| Normal Sample 2 | Heatmap distribution is scattered uniformly across the chest without strong abnormal focus regions.                      |
| Normal Sample 3 | Minimal activation inside lung fields suggests absence of disease-related features.                                      |
| Normal Sample 4 | Model attention remains low-intensity and broadly distributed, which aligns with normal chest anatomy.                   |
| Normal Sample 5 | No major localized saliency regions are observed, indicating the model correctly identifies non-pathological structures. |

---

## C. Correct Viral Pneumonia Predictions

| Sample                   | Observation                                                                                                             |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Viral Pneumonia Sample 1 | Grad-CAM highlights widespread inflammatory regions across both lungs, consistent with viral pneumonia characteristics. |
| Viral Pneumonia Sample 2 | Strong activations are observed in dense pulmonary opacity regions, demonstrating pathology-focused attention.          |
| Viral Pneumonia Sample 3 | Heatmap covers extensive lower lung areas, suggesting recognition of diffuse infection patterns.                        |
| Viral Pneumonia Sample 4 | Model attention is concentrated around asymmetric infiltrates visible in the lung fields.                               |
| Viral Pneumonia Sample 5 | Broad activation patterns indicate the model successfully captured pneumonia-associated radiographic abnormalities.     |

---

# D. Misclassified Samples

| Actual          | Predicted       | Observation                                                                                                                     |
| --------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| COVID           | Viral Pneumonia | Overlapping pulmonary opacities between COVID and viral pneumonia likely caused confusion due to similar radiographic patterns. |
| COVID           | Normal          | Infection regions appear subtle with weak contrast, causing the model to miss disease-specific features.                        |
| COVID           | Viral Pneumonia | Diffuse bilateral infiltrates resemble generalized viral pneumonia characteristics, leading to misclassification.               |
| Normal          | COVID           | Noise and localized high activations may have falsely resembled pathological opacities.                                         |
| Normal          | Viral Pneumonia | Minor imaging artifacts and texture irregularities possibly triggered abnormal feature detection.                               |
| Viral Pneumonia | COVID           | Significant overlap exists between viral pneumonia inflammation and COVID infection patterns in chest X-rays.                   |
| COVID           | Normal          | Grad-CAM shows weak pathological localization, indicating low-confidence feature extraction.                                    |
| Normal          | COVID           | Bright regions outside primary lung structures may have influenced incorrect activation behavior.                               |
| Viral Pneumonia | Normal          | Mild pneumonia manifestations may not have produced sufficiently strong discriminative features.                                |
| COVID           | Viral Pneumonia | Attention maps focus on similar bilateral lung regions common to both respiratory diseases.                                     |

---

# Overall Explainability Discussion

The Grad-CAM analysis demonstrated that the ResNet50 model generally focused on clinically relevant lung regions while making predictions. Correctly classified COVID and Viral Pneumonia samples showed concentrated activations around pulmonary infiltrates and opacity regions, whereas Normal samples exhibited weaker and more diffuse attention maps. Misclassified samples revealed significant overlap between COVID and Viral Pneumonia radiographic characteristics, highlighting the inherent difficulty of distinguishing respiratory infections using chest X-rays alone.

Several failure cases exhibited weak or dispersed activations, indicating that subtle pathological manifestations, low image contrast, and imaging artifacts contributed to incorrect predictions. Importantly, the Grad-CAM visualizations confirmed that the model primarily attended to anatomically meaningful lung structures instead of irrelevant background regions, supporting the reliability and interpretability of the classification pipeline.

# Reason for Freezing ViT-B/16 Backbone During Training

The Vision Transformer (ViT-B/16) architecture contains a significantly larger number of trainable parameters compared to conventional convolutional neural networks such as ResNet50. Full fine-tuning of the transformer backbone requires substantial computational resources, memory, and training time, particularly on laptop GPUs.

To make the training process computationally efficient while preserving the explainability objectives of the project, the pretrained transformer backbone was frozen and only the final classification head was trained. This transfer learning strategy allows the model to retain the generalized visual representations learned from ImageNet while adapting the classifier to the chest X-ray dataset.

The primary goal of the internship project is explainability analysis rather than achieving marginal improvements in classification accuracy. Therefore, reducing training complexity while maintaining strong predictive performance was considered an appropriate design decision.

This approach provided several advantages:

- Reduced training time and GPU memory consumption.
- Improved training stability on limited hardware resources.
- Prevention of excessive overfitting on a relatively small medical imaging dataset.
- Preservation of pretrained semantic feature representations.
- Faster experimentation for explainability analysis using Attention Rollout.

The frozen-backbone strategy enabled efficient comparison between CNN-based Grad-CAM explanations and transformer-based attention visualizations within practical computational constraints.
