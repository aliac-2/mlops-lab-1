# MLOps Lab 1 - Questions and Answers

## Question 1

### Observe the files created by `uv init`. What do you think they contain?

The `uv init` command created the basic structure of the Python project:

- `pyproject.toml`: contains the project metadata, Python version requirements, and project dependencies.
- `.python-version`: specifies the Python version used by the project.
- `README.md`: contains documentation and information about the project.
- `src/`: contains the Python source code of the project.

These files define the Python project structure and help make the environment reproducible.

---

## Question 2

### What are the created files? What do you think they are used for? And which ones should be pushed to Git?

Running `dvc init` creates DVC configuration files, mainly:

- `.dvc/config`: contains the DVC project configuration, such as the configured data remote.
- `.dvc/.gitignore`: prevents DVC internal cache and temporary files from being tracked by Git.
- `.dvcignore`: tells DVC which files or folders it should ignore.

The DVC configuration and metadata files should be pushed to Git so that other users can reproduce the same DVC setup.

The DVC cache, temporary files, and actual dataset files should not be pushed to Git.

## Question 3

### Where are the credentials stored? What are the options other than `--global`? Should the credentials be pushed to GitHub?

The credentials configured using `--global` are stored in the user's global DVC configuration outside the Git repository.

Other configuration scopes include:
- Project configuration, when no scope option is specified.
- `--local`, which stores configuration locally for the current repository and is not tracked by Git.
- `--system`, which stores configuration system-wide.

Credentials such as passwords and access tokens should never be pushed to GitHub because they are secret information.

---

## Question 4

### Take a look at the `.gitignore` file. Explain what happened.

After running `dvc add data`, DVC added:

`/data`

to the `.gitignore` file.

This prevents Git from tracking the actual dataset because the data is now managed by DVC. Git only tracks the DVC metadata and pointer files.

---

## Question 5

### Do you see a `.dvc` file? What does it contain?

Yes. A `data.dvc` file was created.

It contains metadata about the tracked `data` directory, including:

- The MD5 hash of the dataset version.
- The total size of the data.
- The number of files.
- The path to the tracked directory.

In our case, DVC detected 16,643 files and tracks the `data` directory.

The `data.dvc` file acts as a pointer to the actual dataset managed by DVC.


## Question 6

### Check the main branch on GitHub. Is the code there? Is the data there? Do you have any file that points to the data location? And what about DagsHub, do you see the data?

Yes, the code and DVC metadata files are available on GitHub.

The actual dataset is not stored directly in GitHub because the `data` directory is ignored by Git.

The `data.dvc` file is stored in GitHub and acts as a pointer to the version of the dataset tracked by DVC.

After running `dvc push`, the actual data is uploaded to the DagsHub DVC remote and can be accessed from DagsHub.

## Question 7

After cloning the GitHub repository into a new folder, the actual data is not downloaded by Git because the `data` directory is managed by DVC.

The command needed to retrieve the tracked data is:

`dvc pull`

This downloads the data version referenced by `data.dvc` from the configured DVC remote.

## Question 8

No. After checking out the older Git commit and running `dvc checkout`, the `food11_processed` and `food11_processed_mini` folders are no longer present.

This happens because the older `data.dvc` file points to the earlier version of the data that only contained `food11_raw`.