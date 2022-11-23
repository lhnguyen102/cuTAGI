#!/bin/bash
## -----------------
## Check for root/sudo
## -----------------

set -e
set -x

-c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
export PATH="/usr/local/opt/python/libexec/bin:$PATH"
export PATH=/usr/local/bin:/usr/local/sbin:$PATH
brew install pyenv
pyenv init
pyenv install -v ${PYTHON_VERSION_MACOS}
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
echo $PYENV_ROOT
if [[ $GITHUB_ACTIONS ]]; then
    # Set paths for subsequent steps, using ${CUDA_PATH}
    echo "Adding PYTHON to PATH"
    echo "$PYENV_ROOT/bin" >> $GITHUB_PATH
fi