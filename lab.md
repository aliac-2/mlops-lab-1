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


# MLOps Lab 3 - Questions and Answers

## Question 1

### What version number was your model given? What's the difference between a run's logged model artifact and a registered model?

The model was registered under the name `food11` and was given version **1**.

A run's logged model artifact is the model produced and stored as an artifact belonging to a specific MLflow run. A registered model is a named model in the MLflow Model Registry that has its own version numbers and can be managed independently from the run that produced it.

---

## Question 2

### What aliases replaced the old built-in stages in MLflow? Why version a model separately from the run that produced it, and why is an alias more flexible than a fixed stage name?

MLflow uses aliases such as `champion` and `challenger` instead of the old built-in stages such as `Staging` and `Production`.

Versioning a model separately from the run allows multiple versions of the same registered model to be managed independently from the experiments that produced them.

An alias is more flexible because it can be moved from one model version to another without changing the version number itself. For example, the `champion` alias can point to version 1 and later be reassigned to version 2 without changing either version.

---

## Question 3

### Why load the model through an MLflow model URI (`models:/food11@champion`) instead of pointing directly at the `.pth` file on disk? What would you have to change to serve a newer model version?

Using the MLflow model URI allows the application to load the model through the MLflow Model Registry instead of depending on a specific `.pth` file path. This makes model serving independent from the physical location of the model file and allows MLflow to manage the model version and metadata.

To serve a newer model version, the `champion` alias can be moved to the newer registered model version. The serving code can continue using `models:/food11@champion` without changing the model-loading code.

---

## Question 4

### Why copy `pyproject.toml`/`uv.lock` and run `uv sync` before copying the rest of the source code, instead of copying everything at once? What happens to the build cache when you only change a line in `serve.py`?

The dependency files are copied and `uv sync` is run before copying the source code so that the dependency installation layer can be cached separately from the application source code.

If only a line in `serve.py` changes, Docker can reuse the cached dependency layers and only rebuild the layers affected by the source-code change. This makes subsequent builds faster.

---

## Question 5

### What's the size difference between a naive single-stage image and your multi-stage one? Use `docker history <image>` to see which layers are the biggest.

Our multi-stage Docker image `food11-api:latest` has a disk usage of approximately **1.99 GB**.

The largest layer shown by `docker history` is the copied Python virtual environment:

`COPY /app/.venv /app/.venv` — approximately **1.43 GB**.

The current image also contains the Python/Debian base image and system packages.

An exact numerical size difference from a naive single-stage image was not measured because we did not build a separate naive single-stage image. Therefore, no exact single-stage size difference is claimed.

---

## Question 6

### What happens to build speed and image size if you forget the `.dockerignore`? Which of the excluded folders would actually break the build if they were sent to the Docker daemon?

Without `.dockerignore`, unnecessary files such as `.venv/`, `data/`, `lab2_data/`, `mlruns/`, `.git/`, and Python cache files would be included in the Docker build context. This can make sending the build context slower and use more disk space.

With the current Dockerfile, these folders would not directly break the build because the Dockerfile only copies the required dependency files and the `src/` directory. However, sending large datasets, MLflow artifacts, and the local virtual environment is unnecessary and wastes resources.

---

## Question 7

### Why can't the container simply use `127.0.0.1:5000` to reach the MLflow server on your host? What does `host.docker.internal` resolve to?

Inside a Docker container, `127.0.0.1` refers to the container itself, not to the host machine. Therefore, `127.0.0.1:5000` would look for an MLflow server running inside the container.

On Docker Desktop for Windows, `host.docker.internal` is a special hostname that allows a container to reach services running on the host machine. Therefore, we used:

`http://host.docker.internal:5000`

to connect the containerized API to the MLflow server running on the host.

---

## Question 8

### Stop the container and start a new one from the same image. Does the model still load correctly without you rebuilding? What does that tell you about what's baked into the image versus fetched at runtime?

Yes. We stopped the original container and created a new container from the same `food11-api:latest` image without rebuilding the image.

The new container successfully loaded the model and the health endpoint returned:

`{"status":"ok"}`

This shows that the application code and Python dependencies are contained in the Docker image, while the model is fetched from MLflow at runtime.

In our local setup, the MLflow model artifacts are stored in the local `mlruns` directory, so that directory was mounted into the new container at runtime.

---

## Question 9

### The Dockerfile and image are versioned differently — one lives in git, the other doesn't (yet). What's still missing before another machine (like a CI runner or a Kubernetes cluster) could reliably pull and run the exact image you just built?

The Docker image currently exists only in the local Docker environment. To allow another machine, CI runner, or Kubernetes cluster to pull and run the exact image, the image should be pushed to a container registry such as Docker Hub or GitHub Container Registry.

A specific version tag or image digest should also be used instead of relying only on the mutable `latest` tag so that the exact image version can be identified and reproduced.

