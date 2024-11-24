# ImageNet Data Download Script

## Prerequisites

Before running the downloading script, ensure the following tools are installed

1. **Kaggle CLI** (activate Python environment and install using `pip install kaggle`)

2. **Kaggle API Key**:
   - Go to your Kaggle Account Settings
   - Create and download an API token (`kaggle.json`).

   #### If Running Locally:
   - Save the `kaggle.json` file to the directory `~/.kaggle/`:
     ```bash
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

   #### If Running on a Server:
   Follow these steps to transfer your `kaggle.json` file to the server:
   1. **Locate `kaggle.json` File**:
      - `kaggle.json` file is typically downloaded to your local `Downloads` folder after generating an API token.

   2. **Transfer File to Server**:
      - Use the `scp` command:
        ```bash
        scp ~/Downloads/kaggle.json <username>@<server-ip>:~
        ```
        Replace:
        - `~/Downloads/kaggle.json` with the path to your `kaggle.json` file.
        - `<username>` with your server's username.
        - `<server-ip>` with the server's IP address.

   3. **Move File to Correct Directory**:
      - Log in to your server:
        ```bash
        ssh <username>@<server-ip>
        ```
      - Create the `.kaggle` directory and move the file:
        ```bash
        mkdir -p ~/.kaggle
        mv ~/kaggle.json ~/.kaggle/
        ```
      - Secure the file by setting appropriate permissions:
        ```bash
        chmod 600 ~/.kaggle/kaggle.json
        ```

   4. **Verify Setup**:
      - Test the Kaggle CLI on the server:
        ```bash
        kaggle competitions list
        ```
      - If successful, you should see a list of Kaggle competitions.


## Instructions

Follow these steps to download the ImageNet dataset:

1. **Navigate to `cuTAGI` folder**

2. **Make Script Executable**:
   ```bash
   chmod +x scripts/download_imagenet.sh
   ```

4. **Run Script**:
   ```bash
   scripts/download_imagenet.sh
   ```

4. **Data Location**:
   - ImageNet dataset will be downloaded and extracted into the `imagenet_data` directory.

5. **Convert Validation Dataset for PyTorch**:
   - Navigate to the validation data directory:
     ```bash
     cd ILSVRC/Data/CLS-LOC/val
     ```
   - Run the conversion script (see [here](https://discuss.pytorch.org/t/issues-with-dataloader-for-imagenet-should-i-use-datasets-imagefolder-or-datasets-imagenet/115742/3) for more details):
     ```bash
     wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
     ```
6. **Clean Up Kaggle API Key If Running on Server**:

   For security, remove the `kaggle.json` file from the server after downloading:
    ```bash
    rm -rf ~/.kaggle
    ```
