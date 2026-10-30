
import argparse
import math

from mlflow import MlflowClient

MODEL_NAME = "food11"
METRIC_NAME = "test_accuracy"


def get_accuracy(client: MlflowClient, version: str) -> float:
    model_version = client.get_model_version(
        name=MODEL_NAME,
        version=version,
    )

    if not model_version.run_id:
        raise ValueError(f"Version {version} has no associated run")

    run = client.get_run(model_version.run_id)
    accuracy = run.data.metrics.get(METRIC_NAME)

    if accuracy is None:
        raise ValueError(
            f"Metric '{METRIC_NAME}' missing for version {version}"
        )

    return float(accuracy)


def promote(candidate_version: str, test_size: int) -> None:
    if test_size <= 0:
        raise ValueError("Test size must be positive")

    client = MlflowClient()

    champion = client.get_model_version_by_alias(
        MODEL_NAME,
        "champion",
    )

    champion_version = str(champion.version)

    if candidate_version == champion_version:
        raise ValueError("Candidate is already the champion")

    champion_accuracy = get_accuracy(client, champion_version)
    candidate_accuracy = get_accuracy(client, candidate_version)

    standard_error = math.sqrt(
        champion_accuracy * (1 - champion_accuracy) / test_size
    )

    threshold = champion_accuracy + 2 * standard_error

    print(f"Current champion: Version {champion_version}")
    print(f"Candidate: Version {candidate_version}")
    print(f"Champion accuracy: {champion_accuracy:.4f}")
    print(f"Candidate accuracy: {candidate_accuracy:.4f}")
    print(f"Standard error: {standard_error:.4f}")
    print(f"Two standard errors: {2 * standard_error:.4f}")
    print(f"Required accuracy: {threshold:.4f}")

    if candidate_accuracy >= threshold:
        decision = "PROMOTED"

        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="champion",
            version=candidate_version,
        )

        print("Decision: PROMOTED to champion")
    else:
        decision = "REJECTED"

        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="challenger",
            version=candidate_version,
        )

        print("Decision: REJECTED; assigned challenger alias")

    tags = {
        "quality_gate_decision": decision,
        "champion_accuracy": str(champion_accuracy),
        "candidate_accuracy": str(candidate_accuracy),
        "required_accuracy": str(threshold),
        "standard_error": str(standard_error),
        "test_size": str(test_size),
    }

    for key, value in tags.items():
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=candidate_version,
            key=key,
            value=value,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--version", required=True)
    parser.add_argument("--n", type=int, required=True)

    args = parser.parse_args()

    promote(args.version, args.n)


if __name__ == "__main__":
    main()
