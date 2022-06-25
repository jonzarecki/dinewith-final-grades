# Common

General utils I use for every project

## Start Here

1. `git submodule add https://github.com/jonzarecki/common.git`

2. Copy `environment.yaml`, `Dockerfile`,`.gitignore`, `.pylintrc`, `.pre-commit-config.yaml`, `.gitlint`, `.editorconfig`, `setup.cfg` to the repo's main folder.

3. Add the common submodule to PyCharm's github component.

   - Go to settings, Version Control
   - Add the submodule to the list of tracked folders with your root folder.

4. Install pre-commit hooks
   - Install pre-commit in your **local** environment (`pip install pre-commit`)
   - Run `pre-commit install --install-hooks -t pre-commit -t commit-msg`
   - Hooks are built and tested in `https://github.com/jonzarecki/pre-commit-checks`
