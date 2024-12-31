# Setting Up Weights & Biases (WandB)
This dock shows the steps for setting up WandB

## Prerequisites
- Pip Installed: Make sure pip is available to install packages `pip install wandb`.
- WandB Account: Create an account at wandb.ai if you don’t already have one.

## Setting up Steps
### Step 1: Create a WandB Account
1. Go to [wandb.ai](https://wandb.ai).
2. Click on **Sign Up**.
3. Register using your email, GitHub, or Google account.


### Step 2: Get Your API Key
1. After logging in, click on your profile picture in the top-right corner.
2. Select **User Settings** from the dropdown menu.
3. Scroll down to the **API Keys** section.
4. Copy the provided API key.


### Step 3: Save API Key in `.bashrc`
1. Open your terminal.
2. Edit the `.bashrc` file:
   ```bash
   nano ~/.bashrc
   ```
3. Add the following line at the end of the file:
   ```bash
   export WANDB_API_KEY="your_api_key_here"
   ```
   Replace `your_api_key_here` with the API key you copied.
4. Save the file (Ctrl + O, Enter, Ctrl + X in nano).
5. Apply the changes:
   ```bash
   source ~/.bashrc
   ```

### Step 4: Visualize Your Plots
1. Run your Python script with WandB integration.
2. Open [wandb.ai](https://wandb.ai) in your browser.
3. Navigate to **Projects** and select your project e.g., resnet
4. Explore plots for different metrics

