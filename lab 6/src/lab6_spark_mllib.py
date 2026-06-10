from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JDK_CANDIDATES = [
    Path("C:/Misha/lab6_tools/jdk-17"),
    PROJECT_ROOT / ".tools" / "jdk-17",
]
for local_jdk in JDK_CANDIDATES:
    if not os.environ.get("JAVA_HOME") and (local_jdk / "bin" / "java.exe").exists():
        os.environ["JAVA_HOME"] = str(local_jdk)
        os.environ["PATH"] = str(local_jdk / "bin") + os.pathsep + os.environ.get("PATH", "")
        break

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.clustering import BisectingKMeans, GaussianMixture, KMeans
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    ClusteringEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import (
    OneHotEncoder,
    PCA,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.functions import vector_to_array
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


SEED = 42
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURE_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
PROFILE_CATEGORICAL_COLS = CATEGORICAL_FEATURE_COLS + ["Churn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 6: Spark MLlib clustering and classification")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"))
    parser.add_argument("--outputs", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=6)
    return parser.parse_args()


def ensure_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1000:
        return
    print(f"Downloading dataset to {path}")
    urllib.request.urlretrieve(DATASET_URL, path)


def spark_session() -> SparkSession:
    return (
        SparkSession.builder.appName("Lab6SparkMLlib")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "3g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def prepare_data(spark: SparkSession, data_path: Path, outputs_dir: Path):
    raw = spark.read.csv(str(data_path), header=True, inferSchema=True)
    df = raw.drop("customerID")

    df = df.withColumn(
        "TotalCharges",
        F.when(F.trim(F.col("TotalCharges").cast("string")) == "", None).otherwise(
            F.col("TotalCharges").cast("double")
        ),
    )
    df = df.withColumn(
        "SeniorCitizen",
        F.when(F.col("SeniorCitizen").cast("int") == 1, F.lit("Yes")).otherwise(F.lit("No")),
    )

    missing_before = {
        col: df.filter(F.col(col).isNull() | (F.trim(F.col(col).cast("string")) == "")).count()
        for col in NUMERIC_COLS + PROFILE_CATEGORICAL_COLS
    }

    medians: dict[str, float] = {}
    for col in NUMERIC_COLS:
        median = df.approxQuantile(col, [0.5], 0.001)[0]
        medians[col] = float(median)
    df = df.fillna(medians)

    for col in PROFILE_CATEGORICAL_COLS:
        df = df.withColumn(
            col,
            F.when(
                F.col(col).isNull() | (F.trim(F.col(col).cast("string")) == ""),
                F.lit("Unknown"),
            ).otherwise(F.trim(F.col(col).cast("string"))),
        )

    indexed_cols = [f"{col}_idx" for col in CATEGORICAL_FEATURE_COLS]
    encoded_cols = [f"{col}_ohe" for col in CATEGORICAL_FEATURE_COLS]
    indexers = [
        StringIndexer(
            inputCol=col,
            outputCol=f"{col}_idx",
            handleInvalid="keep",
            stringOrderType="frequencyDesc",
        )
        for col in CATEGORICAL_FEATURE_COLS
    ]
    encoder = OneHotEncoder(
        inputCols=indexed_cols,
        outputCols=encoded_cols,
        dropLast=True,
    )
    numeric_assembler = VectorAssembler(
        inputCols=NUMERIC_COLS,
        outputCol="numeric_raw",
        handleInvalid="keep",
    )
    scaler = StandardScaler(
        inputCol="numeric_raw",
        outputCol="numeric_scaled",
        withMean=True,
        withStd=True,
    )
    feature_assembler = VectorAssembler(
        inputCols=encoded_cols + ["numeric_scaled"],
        outputCol="features",
    )
    pipeline = Pipeline(stages=indexers + [encoder, numeric_assembler, scaler, feature_assembler])
    pipeline_model = pipeline.fit(df)
    features_df = pipeline_model.transform(df).cache()
    row_count = features_df.count()
    feature_dim = int(features_df.select("features").first()["features"].size)

    summary = {
        "dataset": "Telco Customer Churn",
        "source_url": DATASET_URL,
        "rows": row_count,
        "columns_after_customer_id_drop": len(df.columns),
        "numeric_features": NUMERIC_COLS,
        "categorical_features": CATEGORICAL_FEATURE_COLS,
        "profile_categorical_features": PROFILE_CATEGORICAL_COLS,
        "missing_before_imputation": missing_before,
        "numeric_imputation_medians": medians,
        "feature_vector_dimension": feature_dim,
    }
    (outputs_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return features_df, summary


def _training_cost(model: Any) -> float | None:
    try:
        return float(model.summary.trainingCost)
    except Exception:
        return None


def _gmm_log_likelihood(model: Any) -> float | None:
    try:
        return float(model.summary.logLikelihood)
    except Exception:
        return None


def fit_clustering_models(features_df, feature_dim: int, k_values: list[int], outputs_dir: Path):
    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="prediction",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )
    n_rows = features_df.count()
    metrics: list[dict[str, Any]] = []
    best_models: dict[str, Any] = {}
    best_rows: dict[str, dict[str, Any]] = {}

    algorithms = {
        "KMeans": lambda k: KMeans(
            featuresCol="features",
            predictionCol="prediction",
            k=k,
            seed=SEED,
            maxIter=50,
            initMode="k-means||",
        ),
        "BisectingKMeans": lambda k: BisectingKMeans(
            featuresCol="features",
            predictionCol="prediction",
            k=k,
            seed=SEED,
            maxIter=50,
        ),
    }

    for name, factory in algorithms.items():
        for k in k_values:
            start = time.perf_counter()
            model = factory(k).fit(features_df)
            train_time = time.perf_counter() - start
            pred = model.transform(features_df).cache()
            silhouette = float(evaluator.evaluate(pred))
            pred.unpersist()
            row = {
                "algorithm": name,
                "k": k,
                "silhouette": silhouette,
                "wcss": _training_cost(model),
                "bic": None,
                "log_likelihood": None,
                "training_time_sec": train_time,
            }
            metrics.append(row)
            if name not in best_rows or silhouette > float(best_rows[name]["silhouette"]):
                best_rows[name] = row
                best_models[name] = model

    for k in k_values:
        start = time.perf_counter()
        model = GaussianMixture(
            featuresCol="features",
            predictionCol="prediction",
            probabilityCol="probability",
            k=k,
            seed=SEED,
            maxIter=40,
            tol=1e-3,
        ).fit(features_df)
        train_time = time.perf_counter() - start
        pred = model.transform(features_df).cache()
        silhouette = float(evaluator.evaluate(pred))
        pred.unpersist()
        log_likelihood = _gmm_log_likelihood(model)
        params_count = (k - 1) + k * feature_dim + k * feature_dim * (feature_dim + 1) / 2
        bic = None
        if log_likelihood is not None:
            bic = float(-2 * log_likelihood + params_count * math.log(n_rows))
        row = {
            "algorithm": "GaussianMixture",
            "k": k,
            "silhouette": silhouette,
            "wcss": None,
            "bic": bic,
            "log_likelihood": log_likelihood,
            "training_time_sec": train_time,
        }
        metrics.append(row)
        if "GaussianMixture" not in best_rows:
            best_rows["GaussianMixture"] = row
            best_models["GaussianMixture"] = model
        else:
            previous = best_rows["GaussianMixture"]
            is_better = False
            if bic is not None and previous["bic"] is not None:
                is_better = bic < previous["bic"]
            elif log_likelihood is not None and previous["log_likelihood"] is not None:
                is_better = log_likelihood > previous["log_likelihood"]
            if is_better:
                best_rows["GaussianMixture"] = row
                best_models["GaussianMixture"] = model

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(outputs_dir / "cluster_metrics.csv", index=False)
    pd.DataFrame(best_rows.values()).to_csv(outputs_dir / "best_cluster_models.csv", index=False)
    plot_clustering_metrics(metrics_df, outputs_dir)
    return best_models, best_rows, metrics_df


def plot_clustering_metrics(metrics_df: pd.DataFrame, outputs_dir: Path) -> None:
    plots_dir = outputs_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    for algorithm in ["KMeans", "BisectingKMeans"]:
        subset = metrics_df[metrics_df["algorithm"] == algorithm]
        plt.figure(figsize=(7, 4.2))
        sns.lineplot(data=subset, x="k", y="silhouette", marker="o", linewidth=2)
        plt.title(f"{algorithm}: silhouette by number of clusters")
        plt.xlabel("Number of clusters, k")
        plt.ylabel("Silhouette")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{algorithm.lower()}_silhouette.png", dpi=180)
        plt.close()

    subset = metrics_df[metrics_df["algorithm"] == "GaussianMixture"].copy()
    plt.figure(figsize=(7, 4.2))
    if subset["bic"].notna().any():
        sns.lineplot(data=subset, x="k", y="bic", marker="o", linewidth=2)
        plt.ylabel("BIC")
        plt.title("Gaussian Mixture: BIC by number of clusters")
    else:
        sns.lineplot(data=subset, x="k", y="log_likelihood", marker="o", linewidth=2)
        plt.ylabel("Log-likelihood")
        plt.title("Gaussian Mixture: log-likelihood by number of clusters")
    plt.xlabel("Number of clusters, k")
    plt.tight_layout()
    plt.savefig(plots_dir / "gaussian_mixture_bic.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.6))
    sns.lineplot(data=metrics_df, x="k", y="silhouette", hue="algorithm", marker="o", linewidth=2)
    plt.title("Silhouette comparison")
    plt.xlabel("Number of clusters, k")
    plt.ylabel("Silhouette")
    plt.tight_layout()
    plt.savefig(plots_dir / "silhouette_comparison.png", dpi=180)
    plt.close()


def choose_cluster_target(best_rows: dict[str, dict[str, Any]]) -> str:
    return max(best_rows, key=lambda name: float(best_rows[name]["silhouette"]))


def profile_clusters(clustered_df, outputs_dir: Path):
    plots_dir = outputs_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    counts_pd = clustered_df.groupBy("cluster").count().orderBy("cluster").toPandas()
    counts_pd.to_csv(outputs_dir / "cluster_counts.csv", index=False)

    numeric_profile = (
        clustered_df.groupBy("cluster")
        .agg(*[F.round(F.avg(col), 3).alias(col) for col in NUMERIC_COLS])
        .orderBy("cluster")
        .toPandas()
    )
    numeric_profile.to_csv(outputs_dir / "cluster_numeric_profile.csv", index=False)

    mode_frames = []
    for col in PROFILE_CATEGORICAL_COLS:
        counts = clustered_df.groupBy("cluster", col).count()
        window = Window.partitionBy("cluster").orderBy(F.desc("count"), F.asc(col))
        top = (
            counts.withColumn("rn", F.row_number().over(window))
            .filter(F.col("rn") == 1)
            .select("cluster", F.lit(col).alias("feature"), F.col(col).alias("mode"), "count")
        )
        mode_frames.append(top)
    modes_df = mode_frames[0]
    for part in mode_frames[1:]:
        modes_df = modes_df.unionByName(part)
    modes_pd = modes_df.orderBy("cluster", "feature").toPandas()
    modes_pd.to_csv(outputs_dir / "cluster_categorical_modes.csv", index=False)

    heatmap_data = numeric_profile.set_index("cluster")[NUMERIC_COLS]
    normalized = (heatmap_data - heatmap_data.mean()) / heatmap_data.std(ddof=0)
    plt.figure(figsize=(7, 4.3))
    sns.heatmap(normalized, annot=heatmap_data, fmt=".1f", cmap="viridis", cbar_kws={"label": "z-score"})
    plt.title("Numeric feature means by cluster")
    plt.xlabel("Feature")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(plots_dir / "cluster_numeric_heatmap.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.3))
    sns.barplot(data=counts_pd, x="cluster", y="count", color="#4f8bc9")
    plt.title("Cluster sizes")
    plt.xlabel("Cluster")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.savefig(plots_dir / "cluster_sizes.png", dpi=180)
    plt.close()

    return counts_pd, numeric_profile, modes_pd


def plot_pca(clustered_df, outputs_dir: Path) -> dict[str, float]:
    plots_dir = outputs_dir / "plots"
    pca_model = PCA(k=2, inputCol="features", outputCol="pca_features").fit(clustered_df)
    pca_df = pca_model.transform(clustered_df.select("cluster", "features")).select(
        "cluster", "pca_features"
    )
    pca_pd = pca_df.toPandas()
    pca_pd["PC1"] = pca_pd["pca_features"].apply(lambda vec: float(vec[0]))
    pca_pd["PC2"] = pca_pd["pca_features"].apply(lambda vec: float(vec[1]))
    pca_pd = pca_pd.drop(columns=["pca_features"])
    pca_pd.to_csv(outputs_dir / "pca_projection.csv", index=False)

    plt.figure(figsize=(8, 5.2))
    sns.scatterplot(
        data=pca_pd,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette="tab10",
        s=20,
        alpha=0.7,
        linewidth=0,
    )
    plt.title("2D PCA projection of clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster", loc="best")
    plt.tight_layout()
    plt.savefig(plots_dir / "pca_clusters.png", dpi=180)
    plt.close()

    variance = [float(x) for x in pca_model.explainedVariance.toArray()]
    return {"pc1": variance[0], "pc2": variance[1], "total": sum(variance)}


def evaluate_predictions_pd(predictions_pd: pd.DataFrame) -> dict[str, float]:
    y_true = predictions_pd["label"].astype(float).to_numpy()
    y_pred = predictions_pd["prediction"].astype(float).to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
    }


def confusion_matrix_pd(predictions_pd: pd.DataFrame) -> pd.DataFrame:
    labels = sorted(predictions_pd["label"].astype(float).unique().tolist())
    matrix = confusion_matrix(
        predictions_pd["label"].astype(float),
        predictions_pd["prediction"].astype(float),
        labels=labels,
    )
    rows = []
    for i, true_label in enumerate(labels):
        for j, predicted_label in enumerate(labels):
            rows.append(
                {
                    "label": true_label,
                    "prediction": predicted_label,
                    "count": int(matrix[i, j]),
                }
            )
    return pd.DataFrame(rows)


def train_classifiers(labeled_df, outputs_dir: Path):
    plots_dir = outputs_dir / "plots"
    labeled_df = labeled_df.withColumn("row_id", F.monotonically_increasing_id()).cache()
    train_df, test_df = labeled_df.randomSplit([0.7, 0.3], seed=SEED)
    train_df = train_df.cache()
    test_df = test_df.cache()
    train_count = train_df.count()
    test_count = test_df.count()
    classes = sorted([float(row["label"]) for row in labeled_df.select("label").distinct().collect()])
    class_count = len(classes)

    tuning_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1",
    )
    model_specs: list[tuple[str, Any, list[Any]]] = []

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        maxIter=80,
        family="auto",
    )
    lr_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.0, 0.01, 0.1])
        .addGrid(lr.elasticNetParam, [0.0, 0.5])
        .build()
    )
    model_specs.append(("LogisticRegression", lr, lr_grid))

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        seed=SEED,
    )
    rf_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [40, 80])
        .addGrid(rf.maxDepth, [5, 8])
        .build()
    )
    model_specs.append(("RandomForest", rf, rf_grid))

    metric_rows = []
    predictions_by_name: dict[str, pd.DataFrame] = {}
    for name, estimator, grid in model_specs:
        tvs = TrainValidationSplit(
            estimator=estimator,
            estimatorParamMaps=grid,
            evaluator=tuning_evaluator,
            trainRatio=0.8,
            parallelism=2,
            seed=SEED,
        )
        start = time.perf_counter()
        tuned = tvs.fit(train_df)
        train_time = time.perf_counter() - start
        predictions_pd = tuned.transform(test_df).select("label", "prediction").toPandas()
        metrics = evaluate_predictions_pd(predictions_pd)
        best_index = int(max(range(len(tuned.validationMetrics)), key=lambda i: tuned.validationMetrics[i]))
        best_params = {
            param.name: value for param, value in tuned.getEstimatorParamMaps()[best_index].items()
        }
        row = {
            "model": name,
            "accuracy": metrics["accuracy"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall": metrics["weighted_recall"],
            "weighted_f1": metrics["weighted_f1"],
            "training_time_sec": train_time,
            "best_params": json.dumps(best_params, ensure_ascii=False),
        }
        metric_rows.append(row)
        predictions_by_name[name] = predictions_pd

    gbt_start = time.perf_counter()
    gbt_prediction_scores = test_df.select("row_id", "label")
    gbt_best_params: list[dict[str, Any]] = []
    binary_evaluator = BinaryClassificationEvaluator(
        labelCol="binary_label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    for class_label in classes:
        class_token = str(int(class_label)) if float(class_label).is_integer() else str(class_label)
        binary_train = train_df.withColumn(
            "binary_label",
            F.when(F.col("label") == F.lit(class_label), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        gbt = GBTClassifier(
            featuresCol="features",
            labelCol="binary_label",
            predictionCol="binary_prediction",
            seed=SEED,
        )
        gbt_grid = (
            ParamGridBuilder()
            .addGrid(gbt.maxDepth, [3, 5])
            .addGrid(gbt.maxIter, [25])
            .build()
        )
        tvs = TrainValidationSplit(
            estimator=gbt,
            estimatorParamMaps=gbt_grid,
            evaluator=binary_evaluator,
            trainRatio=0.8,
            parallelism=2,
            seed=SEED,
        )
        tuned = tvs.fit(binary_train)
        best_index = int(max(range(len(tuned.validationMetrics)), key=lambda i: tuned.validationMetrics[i]))
        best_params = {
            param.name: value for param, value in tuned.getEstimatorParamMaps()[best_index].items()
        }
        gbt_best_params.append({"class": class_label, "params": best_params})
        score_col = f"score_{class_token}"
        scores = (
            tuned.bestModel.transform(test_df.select("row_id", "features"))
            .select("row_id", vector_to_array("rawPrediction")[1].alias(score_col))
        )
        gbt_prediction_scores = gbt_prediction_scores.join(scores, on="row_id", how="inner")

    gbt_scores_pd = gbt_prediction_scores.toPandas()
    score_cols = [
        f"score_{str(int(class_label)) if float(class_label).is_integer() else str(class_label)}"
        for class_label in classes
    ]
    best_score_index = gbt_scores_pd[score_cols].to_numpy().argmax(axis=1)
    gbt_scores_pd["prediction"] = [classes[idx] for idx in best_score_index]
    gbt_predictions_pd = gbt_scores_pd[["label", "prediction"]]
    gbt_metrics = evaluate_predictions_pd(gbt_predictions_pd)
    metric_rows.append(
        {
            "model": "GBT",
            "accuracy": gbt_metrics["accuracy"],
            "weighted_precision": gbt_metrics["weighted_precision"],
            "weighted_recall": gbt_metrics["weighted_recall"],
            "weighted_f1": gbt_metrics["weighted_f1"],
            "training_time_sec": time.perf_counter() - gbt_start,
            "best_params": json.dumps(gbt_best_params, ensure_ascii=False),
        }
    )
    predictions_by_name["GBT"] = gbt_predictions_pd

    metrics_pd = pd.DataFrame(metric_rows).sort_values("weighted_f1", ascending=False)
    metrics_pd.to_csv(outputs_dir / "classifier_metrics.csv", index=False)
    best_name = str(metrics_pd.iloc[0]["model"])
    confusion_pd = confusion_matrix_pd(predictions_by_name[best_name])
    confusion_pd.to_csv(outputs_dir / "confusion_matrix.csv", index=False)

    matrix = confusion_pd.pivot(index="label", columns="prediction", values="count").fillna(0)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues")
    plt.title(f"Confusion matrix: {best_name}")
    plt.xlabel("Predicted cluster")
    plt.ylabel("True cluster")
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    melted = metrics_pd.melt(
        id_vars=["model"],
        value_vars=["accuracy", "weighted_f1"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(8, 4.6))
    sns.barplot(data=melted, x="model", y="value", hue="metric")
    plt.ylim(0, 1.05)
    plt.title("Classifier quality comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(plots_dir / "classifier_metrics.png", dpi=180)
    plt.close()

    split_summary = {
        "train_rows": train_count,
        "test_rows": test_count,
        "class_count": int(class_count),
        "classes": classes,
        "best_classifier": best_name,
    }
    (outputs_dir / "classification_split.json").write_text(
        json.dumps(split_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics_pd, confusion_pd, split_summary


def write_run_summary(
    outputs_dir: Path,
    dataset_summary: dict[str, Any],
    best_rows: dict[str, dict[str, Any]],
    target_algorithm: str,
    pca_variance: dict[str, float],
    split_summary: dict[str, Any],
) -> None:
    run_summary = {
        "dataset": dataset_summary,
        "best_cluster_models": best_rows,
        "classification_target_algorithm": target_algorithm,
        "pca_explained_variance": pca_variance,
        "classification_split": split_summary,
    }
    (outputs_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    outputs_dir = Path(args.outputs)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "plots").mkdir(parents=True, exist_ok=True)
    ensure_dataset(data_path)

    spark = spark_session()
    spark.sparkContext.setLogLevel("WARN")
    try:
        features_df, dataset_summary = prepare_data(spark, data_path, outputs_dir)
        k_values = list(range(args.k_min, args.k_max + 1))
        best_models, best_rows, _ = fit_clustering_models(
            features_df,
            dataset_summary["feature_vector_dimension"],
            k_values,
            outputs_dir,
        )
        target_algorithm = choose_cluster_target(best_rows)
        selected_model = best_models[target_algorithm]
        clustered_df = (
            selected_model.transform(features_df)
            .withColumnRenamed("prediction", "cluster")
            .cache()
        )
        clustered_df.count()
        profile_clusters(clustered_df, outputs_dir)
        pca_variance = plot_pca(clustered_df, outputs_dir)

        labeled_df = clustered_df.withColumn("label", F.col("cluster").cast("double")).cache()
        metrics_pd, _, split_summary = train_classifiers(labeled_df, outputs_dir)
        write_run_summary(
            outputs_dir,
            dataset_summary,
            best_rows,
            target_algorithm,
            pca_variance,
            split_summary,
        )

        print("Lab 6 finished successfully")
        print(f"Dataset rows: {dataset_summary['rows']}")
        print(f"Feature vector dimension: {dataset_summary['feature_vector_dimension']}")
        print(f"Classification target clusters: {target_algorithm}")
        print("Best cluster models:")
        for name, row in best_rows.items():
            print(f"  {name}: k={row['k']}, silhouette={row['silhouette']:.4f}")
        print("Classifier metrics:")
        print(metrics_pd.to_string(index=False))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
