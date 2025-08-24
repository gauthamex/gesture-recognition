# ✋ Hand Gesture Recognition using CNN (PyTorch)

This project implements a **Convolutional Neural Network (CNN)** in **PyTorch** to classify hand gestures using the **LeapGestRecog dataset**. The model learns to recognize different hand gestures from grayscale images and achieves high accuracy.

---

## 📂 Dataset

We use the **[LeapGestRecog dataset](https://www.idiap.ch/en/dataset/leapgestrecog)** which contains:

* **10 users**
* **10 gesture classes** (e.g., palm, fist, thumb, index, ok, etc.)
* Each gesture has **200 images per user** → \~20,000 images total

Directory structure:

```
leapGestRecog/
│── 00/
│   ├── 01_palm/
│   ├── 02_fist/
│   ├── ...
│── 01/
│   ├── 01_palm/
│   ├── 02_fist/
│   ├── ...
```

---

## ⚙️ Workflow

1. **Load Dataset** – Images loaded in grayscale, resized to `64x64`.
2. **Preprocessing** – Normalization (pixel values scaled to `[0,1]`).
3. **Dataset Split** – 80% training, 20% testing.
4. **PyTorch Dataset & DataLoader** – For efficient training.
5. **CNN Model** –

   * Conv2D → ReLU → MaxPool
   * Conv2D → ReLU → MaxPool
   * Fully Connected Layers + Dropout
6. **Training** – CrossEntropyLoss + Adam optimizer.
7. **Evaluation** – Compute test accuracy.
8. **Save Model** – Trained weights saved as `hand_gesture_model.pth`.

---

## 🧠 Model Architecture

```
Input: 1 x 64 x 64 (grayscale image)

[Conv2D(1→32, kernel=3) + ReLU + MaxPool(2)]
[Conv2D(32→64, kernel=3) + ReLU + MaxPool(2)]
[Flatten]
[FC: 64*14*14 → 128 + ReLU + Dropout(0.5)]
[FC: 128 → num_classes]

Output: Gesture Class
```

---

## 📊 Results

* **Training**: 10 epochs
* **Loss**: decreases steadily
* **Accuracy**: \~95% (varies per run & system)

Example output:

```
Epoch 1, Loss: 102.2345
Epoch 2, Loss: 87.4562
...
Epoch 10, Loss: 34.1278
Training finished.
Test Accuracy: 95.42%
Model saved as hand_gesture_model.pth
```

---

## 🚀 Installation & Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the **LeapGestRecog dataset** and place it in the project folder.

4. Run training:

   ```bash
   python hand_gesture_recognition.py
   ```

5. Trained model will be saved as `hand_gesture_model.pth`.

---

## 📦 Requirements

* Python 3.x
* OpenCV (`cv2`)
* NumPy
* scikit-learn
* PyTorch

---

## 📌 Applications

* Human-computer interaction
* Virtual reality / AR gesture input
* Assistive technologies
* Robotics control via hand gestures

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to fork this repo and submit a pull request.

