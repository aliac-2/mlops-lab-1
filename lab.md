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

Running `dvc init` creates DVC configuration and metadata files, mainly:

- `.dvc/config`: contains the DVC project configuration, such as the configured data remote.
- `.dvc/.gitignore`: prevents internal DVC cache and temporary files from being tracked by Git.
- `.dvcignore`: tells DVC which files or folders it should ignore.

The DVC configuration and metadata files should be pushed to Git so that other users can reproduce the same DVC setup.

The DVC cache, temporary files, and actual dataset files should not be pushed directly to Git.

---

## Question 3

### Where are the credentials stored? What are the options other than `--global`? Should the credentials be pushed to GitHub?

Credentials configured using `--global` are stored in the user's global DVC configuration outside the Git repository.

Other configuration scopes include:

- Project/repository configuration when no scope option is specified.
- `--local`, which stores configuration locally for the current repository and is not tracked by Git.
- `--system`, which stores configuration system-wide.

Credentials such as passwords and access tokens should never be pushed to GitHub because they are secret information.

---

## Question 4

### Take a look at the `.gitignore` file. Explain what happened.

After running `dvc add data`, DVC added:

`/data`

to the `.gitignore` file.

This prevents Git from tracking the actual contents of the `data` directory because the data is now managed by DVC. Git tracks the DVC metadata and pointer files instead.

---

## Question 5

### Do you see a `.dvc` file? What does it contain?

Yes. A `data.dvc` file was created.

It contains metadata about the tracked `data` directory, including:

- The hash of the tracked data version.
- The total size of the data.
- The number of files.
- The path to the tracked directory.

The `data.dvc` file does not contain the dataset itself. It acts as a pointer to the specific version of the data managed by DVC.

---

## Question 6

### Check the main branch on GitHub. Is the code there? Is the data there? Do you have any file that points to the data location? And what about DagsHub, do you see the data?

Yes, the code and DVC metadata files are available on GitHub.

The actual contents of the `data` directory are not stored directly in GitHub because the directory is ignored by Git and managed by DVC.

The `data.dvc` file is stored in GitHub and acts as a pointer to the version of the data tracked by DVC.

After running `dvc push`, the DVC-tracked data was successfully uploaded to the DagsHub remote.

---

## Question 7

### In a completely new temporary folder, clone your GitHub repository. Do you see the data folder? What DVC command is needed to get the data folder?

A normal `git clone` retrieves the Git-tracked project files and the `data.dvc` pointer, but it does not retrieve the actual DVC-managed data.

The command needed to retrieve the data referenced by `data.dvc` is:

`dvc pull`

This downloads the required data version from the configured DVC remote.

---

## Question 8

### Do you still see the new folders you created, `food11_processed` and `food11_processed_mini`?

No.

After checking out the older Git commit and running `dvc checkout`, the `food11_processed` and `food11_processed_mini` folders were no longer present, and only the earlier data version remained.

This happens because the older `data.dvc` file points to the previous version of the dataset that only contained `food11_raw`.

After switching back to `main` and running `dvc checkout` again, the processed folders were restored.