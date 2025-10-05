.. _dev_guides:

==========================
Github Development Guide
==========================

.. rubric:: Branch Naming

1. **feature/**: For developing new features.
2. **bugfix/**: To fix bugs in the code. Often created associated to an issue.
3. **hotfix/**: To fix critical bugs in the production.
4. **release/**: To prepare a new release, typically used to do tasks such as last touches and revisions.
5. **docs/**: Used to write, modify or correct documentation.

.. rubric:: Rebasing

Follow these steps to rebase your branch onto the latest version of the ``main`` branch.

1. **Fetch the latest updates from the remote repository:**

   .. code-block:: sh

      git fetch

2. **Switch to the branch you want to rebase:**

   .. code-block:: sh

      git checkout your_branch

3. **Rebase your branch onto the latest version of:** ``main``

   .. code-block:: sh

      git rebase origin/main

4. **Resolve any conflicts if any:**

   - Manually resolve conflicts in files.
   - After resolving, stage the changes:

     .. code-block:: sh

        git add .

5. **Continue the rebase process:**

   .. code-block:: sh

      git rebase --continue

6. **Verify the rebasing:**

   - Once you see the message confirming a successful rebase, your branch is now rebased onto ``main``.

.. rubric:: Code Quality and Pre-Commit Hooks

**Mandatory Pre-Commit Checks (Required for PR)**

To ensure your code meets quality standards and passes all Continuous Integration (CI) checks, you **must** install and run the pre-commit hooks locally before pushing your changes for a Pull Request.

1. **Install pre-commit**:

   .. code-block:: sh

      pip install pre-commit

2. **Install the git hooks into your repository**:

   .. code-block:: sh

      pre-commit install

3. **Manually run hooks against all files (for initial setup and cleanup)**:

   .. code-block:: sh

      pre-commit run --all-files

   .. note:: After installation, the hooks will run automatically on every ``git commit``.

.. rubric:: Locally Compile documentation

To compile the documentation locally, follow these steps:

1. **Install Sphinx and necessary extensions**:

   .. code-block:: sh

      pip install -r docs/requirements_docs.txt

2. **Install pandoc markup converter**:

   - On macOS, you can use Homebrew:

     .. code-block:: sh

        brew install pandoc

   - On Ubuntu/Debian, you can use apt-get:

     .. code-block:: sh

        sudo apt-get install pandoc

3. **Build the documentation**:

   .. code-block:: sh

        sphinx-build -b html docs docs/_build/html

4. **View the documentation**:
   Open the generated HTML files in your web browser:

   - On macOS, you can use the `open` command:

   .. code-block:: sh

      open docs/_build/html/index.html

   - On Linux, you can use `xdg-open`:

   .. code-block:: sh

      xdg-open docs/_build/html/index.html

.. rubric:: Tips and Tools

**Enable Git Autocomplete on macOS**

1.  Add the following command to ``~/.zshrc``:

    .. code-block:: bash

        autoload -Uz compinit && compinit

2.  Activate the changes:

    .. code-block:: bash

        source ~/.zshrc

**Run Memory Check on macOS**

.. code-block:: bash

    leaks --atExit -- bin/run_tests

**Code Formatting in VS Code**

To maintain code consistency, add the following settings to your ``.vscode/settings.json``:

.. code-block:: json

    {
        "C_Cpp.clang_format_fallbackStyle": "{ BasedOnStyle: Google, IndentWidth: 4, ColumnLimit: 80 }",
        "editor.rulers": [80],
        "editor.formatOnSave": true,
        "python.formatting.provider": "black",
        "python.formatting.blackArgs": ["--line-length", 80],
        "editor.trimAutoWhitespace": true,
        "files.trimTrailingWhitespace": true,
        "C_Cpp.errorSquiggles": "disabled"
    }
