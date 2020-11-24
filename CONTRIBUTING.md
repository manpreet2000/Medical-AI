# Contributing Guidelines :

### Best Practices for reporting or requesting for Issues/Enhancements:
  - Follow the Issue Template while creating the issue.
  - Include Screenshots if any (specially for UI related issues)
  - For UI enhancements or workflows, include mockups to get a clear idea.

### Best Practices for assigning an issue:
- If you would like to work on an issue, inform in the issue ticket by commenting on it.
- Please be sure that you are able to reproduce the issue, before working on it. If not, please ask for clarification by commenting or asking the issue creator.

Note: Please do not work on issues which is already being worked on by another contributor. We don't encourage creating multiple pull requests for the same issue. Also, please allow the assigned person some days to work on the issue ( The time might vary depending on the difficulty). If there is no progress after the deadline, please comment on the issue asking the contributor whether he/she is still working on it. If there is no reply, then feel free to work on the issue.

  ### Commits in your pull-requests should

- Have a useful description .

## Advice on pull requests

Pull requests are the easiest way to contribute changes to git repos at Github.
They are the preferred contribution method, as they offer a nice way for
commenting and amending the proposed changes.

- You need a local "fork" of the Github repo.
- Keep your fork up to date
  - Before you create a branch for new changes you should update your fork with the latest changes from our master.

- Use a "feature branch" for your changes ( "ui branch" for UI/UX related changes). That separates the changes in the
  pull request from your other changes and makes it easy to edit/amend commits in the pull request. Workflow using "feature_x" as the example:
  - Update your local git fork to the tip (of the master, usually)
  - Create the feature branch with `git checkout -b feature_x`
  - Edit changes and commit them locally
  - Push them to your Github fork by `git push -u origin feature_x`. That
    creates the "feature_x" branch at your Github fork and sets it as the
    remote of this branch
  - When you now visit Github, you should see a proposal to create a pull
    request

- If you later need to add new commits to the pull request, you can simply
  commit the changes to the local branch and then use `git push` to
  automatically update the pull request.

- If you need to change something in the existing pull request (e.g. to add a
  missing signed-off-by line to the commit message), you can use `git push -f`
  to overwrite the original commits. That is easy and safe when using a feature
  branch. Example workflow:
  - Checkout the feature branch by `git checkout feature_x`
  - Edit changes and commit them locally. If you are just updating the commit
    message in the last commit, you can use `git commit --amend` to do that
  - If you added several new commits or made other changes that require
    cleaning up, you can use `git rebase -i HEAD~X` (X = number of commits to
    edit) to possibly squash some commits
  - Push the changed commits to Github with `git push -f` to overwrite the
    original commits in the "feature_x" branch with the new ones. The pull
    request gets automatically updated

## If you have commit access

- Do NOT use git push --force.
- Do NOT commit to other maintainer's packages without their consent.
- Use Pull Requests if you are unsure and to suggest changes to other
  maintainers.
  



>                                                        Thank you for contributing to this repository! 🙇‍♂️
