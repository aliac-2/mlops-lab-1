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

Running `dvc init` creates the DVC project structure, including `.dvc/config`, `.dvc/.gitignore`, and `.dvcignore`.

The `.dvc/config` file stores project-level DVC settings. The remote storage configuration is added later using `dvc remote add`.


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

---

# Lab 2 - Model Training and Experiment Tracking

## Question 1: What changed in pyproject.toml and uv.lock?

`pyproject.toml` was updated with four new dependencies:
mlflow, torch, torchvision, and scikit-learn.

We also configured a CPU-only PyTorch index.

`uv.lock` was updated with the exact resolved package
versions and their dependencies to make installations reproducible.

## Question 2: What are the backend store and artifact root?

`--backend-store-uri sqlite:///mlflow.db` configures the SQLite database used to store MLflow metadata, including experiments, runs, parameters, metrics, and artifact locations.

`--default-artifact-root ./mlruns` specifies the default location for storing artifacts, such as trained models and output files, for experiments created with that artifact location.

Metadata describes a training run and records its results, while artifacts are the actual files produced by the run.

### Question 3: Why shouldn't mlflow.db and mlruns/ be tracked by Git or DVC?

`mlflow.db` and `mlruns/` contain local MLflow tracking data and artifacts generated during experiments.

They should not be tracked by Git because they can change frequently and contain large binary files, creating unnecessary repository history.

They should not be tracked by DVC either because MLflow already manages experiment metadata, metrics, and artifacts. Tracking these local MLflow outputs again with DVC would duplicate their management.

Git is used to version our training code, DVC is used to version our datasets, and MLflow is used to track our training experiments.

### Question 4: What happens when set_experiment is called with a new name?

When `mlflow.set_experiment("food11")` is called for the first time, MLflow automatically creates a new experiment because it does not exist yet. It assigns the experiment a unique ID and makes it the active experiment for subsequent runs in that process.

In our case, MLflow created the `food11` experiment with ID `2`. The experiment is now visible in the MLflow UI.

### Question 5: What is the difference between mlflow.log_param and mlflow.log_metric? Why does log_metric take a step argument?

`mlflow.log_param` records fixed training settings, such as the learning rate, batch size, and number of epochs.

`mlflow.log_metric` records numerical results, such as training loss and validation accuracy, which can change during training.

The `step` argument identifies the epoch or iteration associated with each metric value, allowing MLflow to plot its evolution. Parameters do not require a step because they are fixed for the run.


### Question 6: Where does the model artifact actually live on disk?

MLflow stores parameters and metrics as run metadata in the SQLite backend (`mlflow.db`), while model artifacts are stored as files.

In our experiment, the run artifact URI is:

`C:/Users/Ali/Desktop/mlops-lab-1/mlruns/2/8f35db59c5024ea3ae9e92e8f1a9dd80/artifacts`

However, MLflow 3 stores our logged model separately, at:

`C:/Users/Ali/Desktop/mlops-lab-1/mlruns/2/models/m-43b42fa9ccd044b4b336861cd0544db7/artifacts`

The model's exact location was verified using `mlflow.get_logged_model()`.
The artifact location above belongs to our initial one-epoch test run. MLflow 3 stores logged models separately from ordinary run artifacts.

The model associated with our selected five-epoch run can also be inspected through the MLflow UI under the logged model's Artifacts tab, which contains the saved model data, `MLmodel`, and environment files.


### Question 7: Which learning rate gave the best validation accuracy? Is higher always better?

Among the tested learning rates, 0.01 achieved the highest final validation accuracy of 67.88%, compared with 66.97% for 0.001 and 38.96% for 0.0001, using batch size 32 and 5 epochs.

A higher learning rate is not always better. An excessively high learning rate can make optimization unstable, while a very small learning rate may require more epochs to converge. Our results only establish the comparison for the tested settings.

### Question 8: What pattern do you see in the parallel coordinates comparison?

With batch size fixed at 32, increasing the learning rate from 0.0001 to 0.001 substantially improved final validation accuracy, while increasing it from 0.001 to 0.01 produced a smaller improvement.

With learning rate fixed at 0.001, increasing batch size from 32 to 64 slightly reduced final validation accuracy from 66.97% to 66.51%. These observations are limited to our four runs.

### Question 9: Which run has the highest validation accuracy? What is its run ID?

The run with learning rate 0.01, batch size 32, and 5 epochs achieved the highest final validation accuracy of 67.88% among our four runs.

Run ID: `7c2f2a7aa7344fb8bfeb26c2f9d1dd3c`

Its final test accuracy was 72.54%. We will retain this run ID for Lab 3.

