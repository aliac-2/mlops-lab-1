# MLOps Lab 1 - Git/DVC and Data Preparation

## Question 1

### Observe the files created by `uv init`. What do you think they contain?

The `uv init` command created the basic structure of the Python project:

- `pyproject.toml`: contains the project metadata, Python version requirements, and project dependencies.
- `.python-version`: specifies the Python version used by the project.
- `README.md`: contains documentation and information about the project.
- `src/`: contains the Python source code of the project.

These files define the Python project structure and make the environment reproducible.

## Question 2

### What are the files created by DVC? What are they used for? Which ones should be pushed to Git?

Running `dvc init` creates DVC configuration files, mainly:

- `.dvc/config`: contains the DVC project configuration, such as the configured data remote.
- `.dvc/.gitignore`: prevents internal DVC cache and temporary files from being tracked by Git.
- `.dvcignore`: tells DVC which files or folders it should ignore.

The DVC configuration and metadata files should be pushed to Git so that other developers can use the same DVC setup.

The actual DVC cache, temporary files, and dataset files should not be pushed to Git.