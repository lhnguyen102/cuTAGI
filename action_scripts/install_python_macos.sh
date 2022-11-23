#!/bin/bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
export PATH="/usr/local/opt/python/libexec/bin:$PATH"
export PATH=/usr/local/bin:/usr/local/sbin:$PATH
brew install pyenv
pyenv install ${PYTHON_VERSION_MACOS}
if [[ $GITHUB_ACTIONS ]]; then
    # Set paths for subsequent steps, using ${CUDA_PATH}
    echo "Adding PYTHON to PATH"
    echo "$/usr/local/opt/python/libexec/bin" >> $GITHUB_PATH
    echo "$/usr/local/bin:/usr/local/sbin" >> $GITHUB_PATH
fi