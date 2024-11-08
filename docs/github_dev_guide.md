# Github Development Guide

## Branch Naming
1. **feature/**: For developing new features,
2. **bugfix/**: To fix bugs in the code. Often created associated to an issue.
3. **hotfix/**: To fix critical bugs in the production.
4. **release/**: To prepare a new release, typically used to do tasks such as last touches and revisions.
5. **docs/**: Used to write, modify or correct documentation.

## Rebasing

Follow these steps to rebase your branch onto the latest version of the `main` branch.

1. **Fetch the latest updates from the remote repository:**
   ```sh
   git fetch
   ```

2. **Switch to the branch you want to rebase:**
   ```sh
   git checkout your_branch
   ```

3. **Rebase your branch onto the latest version of `main`:**
   ```sh
   git rebase origin/main
   ```

4. **Resolve any conflicts if any:**
   - Manually resolve conflicts in files.
   - After resolving, stage the changes:
     ```sh
     git add .
     ```

5. **Continue the rebase process:**
   ```sh
   git rebase --continue
   ```

6. **Verify the rebasing:**
   - Once you see the message confirming a successful rebase, your branch is now rebased onto `main`.


## Pull Request Template
### Description

Provide a brief description of the changes introduced by this pull request. Explain the problem you're addressing or the feature you're implementing.

### Changes Made

Provide a detailed explanation of the changes you made in this PR. Include relevant code snippets or screenshots if necessary.

### Related Issue(s)

Cite any related GitHub issues or JIRA tickets that are addressed or resolved by this PR. Use the syntax `Resolves #123` or `Fixes #456`.

### Checklist

- [ ]  I have followed the project's coding conventions and style guidelines.
- [ ]  I have updated the documentation, if applicable.
- [ ]  I have rebased my branch on the latest upstream code to incorporate any changes.
- [ ]  I have tested the changes locally.


### Notes for Reviewers

Provide any additional context or notes that the reviewers should be aware of while reviewing this PR and how to run the test.