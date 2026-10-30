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

The model was registered under the name `food11` and was given version **2**.

A run's logged model artifact is the model produced and stored as an artifact belonging to a specific MLflow run. A registered model is a named model in the MLflow Model Registry that has its own version numbers and can be managed independently from the run that produced it.

---

## Question 2

### What aliases replaced the old built-in stages in MLflow? Why version a model separately from the run that produced it, and why is an alias more flexible than a fixed stage name?

MLflow uses aliases such as `champion` and `challenger` instead of the old built-in stages such as `Staging` and `Production`.

Versioning a model separately from the run allows multiple versions of the same registered model to be managed independently from the experiments that produced them.

An alias is more flexible because it can be moved from one model version to another without changing the version number itself. For example, the `champion` alias can point to one model version and later be reassigned to another version without changing the version numbers.

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

### What's the size difference between a naive single-stage image and your multi-stage one? Use `docker history <image>` to see which layers are the biggest?

The multi-stage Docker image `food11-api:latest` has a disk usage of approximately **1.99 GB**, while the naive single-stage image `food11-api:single-stage` has a disk usage of approximately **2.17 GB**.

Therefore, the multi-stage image is approximately **0.18 GB (180 MB) smaller**, or about **8.3% smaller** than the single-stage image.

The `docker history` output shows that the largest layer in the single-stage image is the virtual environment created by `uv sync`, at approximately **1.48 GB**. The single-stage image also contains an approximately **82.5 MB** layer for installing `uv`.

In the multi-stage image, the virtual environment is copied from the builder stage into the runtime image, with the corresponding layer being approximately **1.43 GB**.

This demonstrates that the multi-stage build avoids keeping unnecessary builder-related layers in the final runtime image.

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

Yes. We stopped the original container and started a new container from the same `food11-api:latest` image without rebuilding the image.

The new container successfully loaded the model and the health endpoint returned:

`{"status":"ok"}`

The prediction endpoint also successfully returned a prediction for an input image.

This shows that the application code and Python dependencies are contained in the Docker image, while the model is loaded from MLflow at runtime.

In our local setup, the MLflow model artifacts are stored in the local `mlruns` directory, so that directory was mounted into the new container at runtime.

---

## Question 9

### The Dockerfile and image are versioned differently — one lives in git, the other doesn't (yet). What's still missing before another machine (like a CI runner or a Kubernetes cluster) could reliably pull and run the exact image you just built?

The Docker image currently exists only in the local Docker environment. To allow another machine, CI runner, or Kubernetes cluster to pull and run the exact image, the image should be pushed to a container registry such as Docker Hub or GitHub Container Registry.

A specific version tag or image digest should also be used instead of relying only on the mutable `latest` tag so that the exact image version can be identified and reproduced.

---

# MLOps Lab 4 — Orchestrating the App with Docker Compose

## Question 1 — What happens without a mounted volume?

Without a mounted volume, everything written to `/mlflow-data` is stored in the container's writable filesystem. Stopping and restarting the same container preserves that data, but removing it deletes the writable layer. A new container therefore starts without the previous MLflow database or artifacts.

## Question 2 — Why use a named volume instead of a bind mount?

A named volume is managed by Docker and persists independently of the container. It is convenient for application data and does not depend on a specific host directory.

A bind mount would also work, but it requires managing the host path and permissions.

## Question 3 — Why does `mlflow` resolve inside Docker Compose?

Docker Compose creates a private network with built-in DNS. Services can communicate using their service names, so `mlflow` resolves to the MLflow container.

In Lab 3, MLflow ran directly on the host machine, which required `host.docker.internal` to reach it from inside Docker.

## Question 4 — Why use `INFERENCE_URL`?

The environment variable allows the frontend to connect to different inference endpoints without changing its source code.

Inside Compose, it uses `http://inference:8000`. Outside Compose, it can use another reachable address.

## Question 5 — Why doesn't inference publish port 8000?

The frontend communicates with the inference service through Docker Compose's internal network using `http://inference:8000`.

Therefore, inference does not need a published host port.

## Question 6 — What happens when MLflow isn't ready?

`depends_on` ensures startup order but does not guarantee readiness.

If inference attempts to load the model before MLflow is ready, application startup may fail and the container may exit. Health checks or retry logic can solve this problem.

## Question 7 — Which services publish ports?

Our `docker compose ps` output confirmed:

- MLflow: `5000:5000` — published.
- Frontend: `8501:8501` — published.
- Inference: `8000/tcp` — internal only.

This matches the Compose configuration.

## Question 8 — What happens after promoting a model?

Initially, the `champion` alias pointed to `food11` Version 2.

We reassigned the alias to Version 1. After restarting inference with `docker compose restart inference`, FastAPI loaded Version 1 instead of Version 2.

Using the same food image, our observed results were:

- Version 2: Class 0, confidence 64.3%.
- Version 1 after restart: Class 0, confidence 32.2%.

Restarting inference allowed it to use the updated model alias.

## Question 9 — Why does restart work without rebuilding?

The Docker image contains the application code and dependencies, while the registered model is loaded from MLflow during application startup.

Restarting the inference container reloads the model currently assigned to the `champion` alias without rebuilding the image.

## Question 10 — What happens with `down` versus `down -v`?

We ran `docker compose down` followed by `docker compose up -d`.

All three services restarted successfully, and MLflow still contained the `food11` model with the `champion` alias assigned to Version 1.

This confirms that the named volume preserved the database and model artifacts.

For the `down -v` test, we used an isolated Compose project with an Alpine container and a temporary named volume to avoid deleting our main MLflow data.

After removing the volume and recreating the container, the test returned:

`FILE_MISSING`

This demonstrates that `down -v` removes Compose-managed volumes and their stored data.

## Question 11 — What would change for scaling and high availability?

Docker Compose is primarily designed for a single host.

To run multiple inference replicas across machines with load balancing and automatic failover, we would use an orchestration platform such as Kubernetes.

For MLflow high availability, we would use an external database such as PostgreSQL and shared artifact storage such as S3 or MinIO, allowing multiple MLflow instances to run behind a load balancer.

## Additional Implementation Notes

Our original MLflow database contained Windows artifact paths, so we created a backup and migrated the database and artifacts into a Docker named volume.

We updated the artifact paths for Linux and shared the MLflow volume with the inference container.

The frontend was adjusted to use the actual FastAPI response fields, `class_id` and `confidence`.

We also corrected the confidence calculation using softmax because the model returned raw logits. The final end-to-end test successfully displayed a prediction with a valid confidence percentage.

## Conclusion

Lab 4 successfully deployed the Food-11 application using Docker Compose with MLflow, FastAPI, and Streamlit. We verified model serving, model version switching, service communication, and persistent storage.


---

# MLOps Lab 5 — CI/CD with GitHub Actions

## Question 1 — Why should unit tests avoid loading the real MLflow model?

Unit tests should be fast, isolated, and deterministic. Loading the real model would introduce external dependencies unrelated to the code being tested.

The tests could fail because the MLflow server is unavailable, because network access or authentication fails, or because the model artifacts cannot be accessed. Loading the real model is also much slower than using a fake model.

We used `SKIP_MODEL=1` to avoid loading MLflow during testing.

## Question 2 — What does the fake-model API test prove, and what does it not prove?

The fake-model test confirms that FastAPI accepts an uploaded image, applies preprocessing, calls the model's `predict()` method, and returns a valid JSON response with `class_id` and a confidence value between 0 and 1.

It also checks that the predicted class is 0, corresponding to Bread.

However, it does not prove that the real trained model is accurate, that MLflow can load model artifacts, or that the actual inference pipeline behaves correctly with the registered model.

Our local tests passed: **5 tests passed**, with four deprecation warnings.

## Question 3 — Which check caught the duplicate, and why is data leakage dangerous?

The **No duplicates across splits** check detected the copied image by comparing MD5 hashes of image contents.

We created a temporary duplicate from `training/Bread` in `evaluation/Bread`. The validator reported `FAIL` for cross-split duplicates.

After removing the duplicate, all five validation checks passed.

A duplicate between training and evaluation is dangerous because the model may be evaluated on images it has already seen, artificially increasing the reported test accuracy. Unlike a corrupted image, which may crash training and expose the issue immediately, data leakage can remain unnoticed and produce misleading results.

The original mini dataset also contained cross-split duplicates. We created and validated a separate cleaned copy without modifying the original DVC-managed data.

## Question 4 — Where should data validation run when CI cannot access DVC data?

Two appropriate places are:

1. In a data preparation or training pipeline after retrieving the dataset from its DVC remote, before training starts.
2. On a self-hosted CI runner or a scheduled validation job that has access to the DVC remote and its credentials.

This ensures the contract is checked even when GitHub-hosted runners cannot access the dataset.

## Question 5 — What does `needs: test` do?

The `smoke` job declares `needs: test`, so it only runs after the `test` job succeeds.

If a unit test fails, the smoke job is skipped. Without `needs`, GitHub Actions could start both jobs independently, wasting resources building a Docker image from code that has already failed tests.

We confirmed this behavior during our intentionally failing CI experiment.

## Question 6 — Compare the first and second CI runs.

The first GitHub Actions run completed successfully in **2 minutes 8 seconds**.

We then pushed an empty commit using `git commit --allow-empty -m "ci"`. The second run also passed and completed in **2 minutes 2 seconds**.

The second run was 6 seconds faster. Dependency caching may have contributed, but we did not measure individual step durations, so we cannot definitively attribute the improvement to one step. Runner scheduling and Docker build time can also affect the total duration.

## Question 7 — Why tag Docker images with the Git commit SHA instead of `latest`?

A commit SHA uniquely identifies the source-code revision used to build an image.

The `latest` tag is mutable. For example, machine A could pull `latest` on Monday, while machine B pulls `latest` after a new build on Tuesday. Both machines would report using `latest`, even though their application code differs.

By using the full commit SHA, we can identify and retrieve the exact image associated with a particular Git revision.

Our image was successfully published to GHCR and downloaded using:

`ghcr.io/aliac-2/mlops-lab-1:55564b5c3a6e175b0300b6e6f201bee3ddb44396`

## Question 8 — Why declare GitHub Actions token permissions explicitly?

The image publishing job requires `packages: write` to upload container images to GitHub Container Registry and `contents: read` to access repository contents.

Declaring these permissions follows the principle of least privilege.

If a compromised workflow step received an excessively powerful token, it might modify repository contents, delete packages, change workflows, or otherwise abuse whatever permissions the token granted.

## Question 9 — What happened in the failing pull request?

We created the branch `lab5-failing-test` and intentionally changed a test expectation from `Bread` to `Pizza`.

GitHub Actions detected the failure, and the dependent smoke job was skipped.

We then corrected the test, pushed a new commit, and confirmed the CI checks passed. The pull request was merged into `main` successfully.

Compared with relying on developers to run tests manually, required CI checks give the team a consistent, automated way to detect regressions before merging code.

## Question 10 — What decisions did the model quality gate make?

Our current champion was registered model **Version 1**, with test accuracy **70.26%**.

We used an evaluation split of **1,096 images** and calculated:

\[
SE = \sqrt{\frac{p(1-p)}{n}}
\]

With champion accuracy \(p = 0.7026\), the standard error was approximately **0.0138**, so \(2SE\) represented **2.76 percentage points**.

The minimum candidate accuracy required for promotion was approximately **73.02%**.

| Candidate | Learning rate | Test accuracy | Gate decision |
|---|---|---|---|
| Version 3 | 0.0001 | 38.77% | Rejected |
| Version 4 | 0.01 | 72.51% | Rejected |

Both candidates failed to reach the threshold.

The `champion` alias remained assigned to Version 1. The `challenger` alias was assigned to the latest rejected candidate, Version 4.

Our `promote.py` script also recorded the decisions and relevant accuracy values as model-version tags in MLflow.

## Question 11 — How could model promotion run automatically?

One solution is to run promotion on a **self-hosted GitHub Actions runner** located on the same network as the MLflow server. That runner would have access to the model registry and could execute the quality gate after training.

A second solution is to move MLflow to an accessible hosted service such as DagsHub or another authenticated remote MLflow deployment. A CI workflow could then call `promote.py` securely using stored credentials.

## Question 12 — Diagram of the CI/CD pipeline

```text
Developer
   |
   | git push / pull request
   v
GitHub Actions                 [AUTOMATIC]
   |
   v
test job
  - Install dependencies
  - Ruff linting
  - Pytest unit/API tests
   |
   | only when successful
   v
smoke job
  - Build Docker image
  - Start FastAPI with SKIP_MODEL=1
  - Probe /health
   |
   | only on main
   v
push job
  - Authenticate to GHCR
  - Publish Docker image
  - Tag image with Git commit SHA   [CODE VERSION]
   |
   v
GHCR image
   |
   | manual pull and deployment
   v
Running inference container
   |
   | load models:/food11@champion at startup
   v
MLflow Model Registry           [MODEL VERSION]
   ^
   |
Training and evaluation         [MANUAL]
   |
   v
promote.py quality gate         [MANUAL TRIGGER]
  - Compare candidate against champion
  - Promote or assign challenger
  - Record decision tags
```

The code version is selected through the Git commit SHA during the automatic image publishing stage.

The model version is selected independently through the MLflow `champion` alias.

CI linting, testing, Docker smoke testing, and publishing to GHCR are automatic. Model training, triggering promotion, and deploying the image remain manual in our current setup.

## Conclusion

Lab 5 successfully introduced automated GitHub Actions testing, Docker smoke testing, and image publishing to GitHub Container Registry. We also implemented dataset validation, demonstrated prevention of data leakage, tested pull request checks, and implemented a statistical model promotion quality gate.

