# 🚀 **Trash Detection Model Centric Track**

Welcome to the Model-Centric Track of the Trash Detection Challenge! 🎉

This track challenges you to push the boundaries of tiny computer vision by designing innovative model architectures for the Trash Detection Usecase.

🔗 Learn More: Trash Detection Challenge Details (Challenge Link)


## 🌟 **Challenge Overview**

Participants are invited to:

1. **Design novel model architectures** to achieve high accuracy.
2. Optimize for **resource efficiency** (e.g., memory, inference time).
3. Evaluate models on a **private test set** of the Trash Detection Dataset.

You can modify the **model architecture** freely

## 🛠️ **Getting Started Without Docker**

### 💻 **Running the scripts**

Run the following command inside the directory where you cloned this repository:

**NOTE:** It is strongly recommended to use **Python 3.10** for optimal compatibility with the packages mentioned.

To install Python 3.10 on your system if it is **not already present**, follow the steps below or skip it:

**On macOS/Linux:**

```bash
sudo apt update

sudo apt install -y software-properties-common

sudo add-apt-repository ppa:deadsnakes/ppa

sudo apt update

sudo apt install python3.10

python3.10 --version

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

sudo update-alternatives --config python3

sudo apt install python3.10-distutils

wget https://bootstrap.pypa.io/get-pip.py

python3.10 get-pip.py

sudo rm /usr/bin/pip

sudo ln -s /home/devuser/.local/bin/pip /usr/bin/pip
```

**On Windows:**

Download [Python3.10](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe) and install it on the system.

**Creating a Python Virtual Environment**

```bash
python3 -m venv <venv-name>
```

- This will create a directory named <venv-name> in your current working directory, containing a self-contained Python environment with its own Python interpreter and libraries.

**Activating the Virtual Environment**

Once the virtual environment is created, activate it as follows:

**On macOS/Linux:**

```bash
source <venv-name>/bin/activate
```

**On Windows:**

```bash
<venv-name>\Scripts\activate
```

**Dataset Download**:

**Dataset Size:** 2.5 GB

```bash
pip install -r requirements.txt && cd data && python3 download_dataset.py
```
- This command will execute a script to download the required dataset. The dataset will be stored in the "data" directory on the host system, making it accessible for subsequent processing or model training.

**Training**:

```bash
bash ../scripts/script.sh
```

- **NOTE**: It is highly recommended to train the model on a GPU, as it is specifically optimized for GPU operations. Training on a CPU may result in significantly longer processing times due to the CPU's limitations in handling the computational demands of model training.
- This will download the dataset and perform the training based on the model present in the script.py.
- The model inside the script can be updated based on the architecture of the model that you choose. Rerun the script once the model file is updated.

**Testing**:

```bash
pip install -r ../requirements.txt && cd ../validation && python3 model_validation.py
```

- This will evaluate the trained TensorFlow Lite model on a test dataset and calculate the average Intersection over Union (IoU) and class accuracy by comparing the predicted bounding boxes and class labels with the ground truth annotations.

---
## **How to Submit**

- Generate the tflite model file using the script given.
- Place the model in the models directory
- Submit the model (the repo link) through the submission tab in the [ModelNova Challenge](https://modelnova.ai/).
- You can submit any number of times before the due date for the challenge.

## 🌟 Monitor Progress

- Check the Leaderboard in the [ModelNova Challenge](https://modelnova.ai/) on a periodic basis to know your results.

---

## 🎯 **Tips for Success**

- **Focus on Data Quality**: Explore label correction, data augmentation, and other preprocessing techniques.
- **Stay Efficient**: The dataset is large, so plan your modifications carefully.
- **Collaborate**: Join the [community discussions](#) to share ideas and tips!

---

## 📚 **Resources**

- [Yolov8 Model Documentation](https://docs.ultralytics.com/models/yolov8/)
- [Docker Documentation](https://docs.docker.com/)
- [Trash Detection Dataset](https://universe.roboflow.com/nora-slimani/trash-detection-otdmj)
- [TACO Dataset](http://tacodataset.org/)

---

## 📞 **Contact Us**

For inquiries, support, or any questions, feel free to reach out through our [Contact Page](https://modelnova.ai/contact-us)

---

🌟 **Happy Innovating and Good Luck!** 🌟
