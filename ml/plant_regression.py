import argparse
import json
import logging
import os
import sys
from datetime import datetime

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


TARGET_COLUMN = "price_rating"
DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "plant_data",
    "plants.csv",
)
POSTGRES_SCHEMA = "rg"
POSTGRES_TABLE = "plant_features"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "plant_price_model.pkl")
MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "plant_price_model_metadata.json")

NUMERIC_SOURCE_COLUMNS = [
    "bloom_start",
    "bloom_end",
    "height_min",
    "height_max",
    "space_min",
    "space_max",
    "bloom_len",
    "height_avg",
    "tolerance_score",
    "wildlife_score",
]

BOOLEAN_SOURCE_COLUMNS = [
    "color_red",
    "color_orange",
    "color_yellow",
    "color_green",
    "color_blue",
    "color_purple",
    "color_pink",
    "color_brown",
    "color_white",
    "flood_tolerant",
    "drought_tolerant",
    "road_salt_tolerant",
    "sun_full",
    "sun_partial",
    "sun_shade",
    "moisture_wet",
    "moisture_med",
    "moisture_dry",
    "benefits_pollinators",
    "benefits_butterflies",
    "benefits_hummingbirds",
    "benefits_birds",
    "deer_resistant",
    "pos_base",
    "pos_slope",
    "pos_margin",
]

CATEGORICAL_SOURCE_COLUMNS = [
    "form",
    "soil_pref",
    "soil_ph",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a rain garden popularity regression model from CSV or PostgreSQL."
    )
    parser.add_argument(
        "--source",
        choices=["csv"],
        default="csv",
        help="Data source to use. CSV is the only supported source.",
    )
    parser.add_argument(
        "--csv-path",
        default=DEFAULT_CSV_PATH,
        help=f"Path to the CSV file when using --source csv. Default: {DEFAULT_CSV_PATH}",
    )
    return parser.parse_args()


def load_csv_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, na_values=["NULL", "null", ""])
    logging.info(f"Loaded CSV file: {csv_path}")
    print(f"Loaded CSV file: {csv_path}")
    return df


def load_postgres_dataframe() -> pd.DataFrame:
    raise SystemExit(
        "PostgreSQL loading has been removed from this backend. "
        "Use --source csv and put your dataset at plant_data/plants.csv or pass --csv-path."
    )


def load_dataframe(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    return load_csv_dataframe(args.csv_path), "CSV"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()

    for column in NUMERIC_SOURCE_COLUMNS:
        if column in engineered.columns:
            engineered[column] = pd.to_numeric(engineered[column], errors="coerce")

    for column in BOOLEAN_SOURCE_COLUMNS:
        if column in engineered.columns:
            engineered[column] = engineered[column].replace(
                {True: 1, False: 0, "True": 1, "False": 0}
            )
            engineered[column] = pd.to_numeric(engineered[column], errors="coerce")

    if {"bloom_start", "bloom_end"}.issubset(engineered.columns):
        engineered["bloom_len"] = engineered["bloom_end"] - engineered["bloom_start"] + 1

    if {"height_min", "height_max"}.issubset(engineered.columns):
        engineered["height_avg"] = (
            engineered["height_min"] + engineered["height_max"]
        ) / 2

    if {"space_min", "space_max"}.issubset(engineered.columns):
        engineered["space_avg"] = (
            pd.to_numeric(engineered["space_min"], errors="coerce")
            + pd.to_numeric(engineered["space_max"], errors="coerce")
        ) / 2

    tolerance_parts = [
        column
        for column in ["flood_tolerant", "drought_tolerant", "road_salt_tolerant"]
        if column in engineered.columns
    ]
    if tolerance_parts and "tolerance_score" not in engineered.columns:
        engineered["tolerance_score"] = (
            engineered[tolerance_parts].fillna(False).astype(int).sum(axis=1)
        )

    wildlife_parts = [
        column
        for column in [
            "benefits_pollinators",
            "benefits_butterflies",
            "benefits_hummingbirds",
            "benefits_birds",
        ]
        if column in engineered.columns
    ]
    if wildlife_parts and "wildlife_score" not in engineered.columns:
        engineered["wildlife_score"] = (
            engineered[wildlife_parts].fillna(False).astype(int).sum(axis=1)
        )

    return engineered


def choose_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric_columns = [
        column for column in [*NUMERIC_SOURCE_COLUMNS, "space_avg"] if column in df.columns
    ]
    boolean_columns = [
        column for column in BOOLEAN_SOURCE_COLUMNS if column in df.columns
    ]
    categorical_columns = [
        column for column in CATEGORICAL_SOURCE_COLUMNS if column in df.columns
    ]
    return numeric_columns, boolean_columns, categorical_columns


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN not in df.columns:
        raise SystemExit(f"Missing target column: {TARGET_COLUMN}")

    cleaned = df.copy()
    cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")
    cleaned = cleaned[cleaned[TARGET_COLUMN].notna()].copy()

    if len(cleaned) < 10:
        raise SystemExit(
            f"Not enough usable rows to train a model. Found {len(cleaned)} rows."
        )

    return cleaned


def build_preprocessor(
    numeric_columns: list[str],
    boolean_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_columns,
            )
        )

    if boolean_columns:
        transformers.append(
            (
                "boolean",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                    ]
                ),
                boolean_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise SystemExit("No usable feature columns were found for modeling.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def print_dataset_summary(
    source_name: str,
    raw_df: pd.DataFrame,
    model_df: pd.DataFrame,
    numeric_columns: list[str],
    boolean_columns: list[str],
    categorical_columns: list[str],
) -> None:
    print("=== Dataset Summary ===")
    print(f"Source: {source_name}")
    print(f"Rows loaded: {len(raw_df)}")
    print(f"Rows with target available: {len(model_df)}")
    print(
        "Features used: "
        f"{len(numeric_columns)} numeric, "
        f"{len(boolean_columns)} boolean, "
        f"{len(categorical_columns)} categorical"
    )
    logging.info(
        f"Dataset: {len(model_df)} rows, "
        f"{len(numeric_columns)} numeric, {len(boolean_columns)} boolean, {len(categorical_columns)} categorical"
    )


def evaluate_model(name: str, pipeline, x_train, x_test, y_train, y_test) -> tuple[dict, Pipeline]:
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics = {
        "name": name,
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
    }
    return metrics, pipeline


def print_linear_coefficients(
    fitted_pipeline: Pipeline,
    numeric_columns: list[str],
    boolean_columns: list[str],
    categorical_columns: list[str],
) -> None:
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    model = fitted_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out(
        numeric_columns + boolean_columns + categorical_columns
    )
    coefficients = pd.Series(model.coef_, index=feature_names).sort_values(
        key=lambda series: series.abs(), ascending=False
    )

    print("\n=== Top Linear Drivers ===")
    for name, value in coefficients.head(5).items():
        print(f"{name}: {value:.4f}")


def print_random_forest_importance(
    fitted_pipeline: Pipeline,
    numeric_columns: list[str],
    boolean_columns: list[str],
    categorical_columns: list[str],
) -> None:
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    model = fitted_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out(
        numeric_columns + boolean_columns + categorical_columns
    )
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(
        ascending=False
    )

    print("\n=== Top Random Forest Drivers ===")
    for name, value in importances.head(5).items():
        print(f"{name}: {value:.4f}")


def print_cross_validation(
    x_values: pd.DataFrame,
    y_values: pd.Series,
    preprocessor: ColumnTransformer,
) -> float | None:
    if len(x_values) < 20:
        return None

    splits = min(5, len(x_values))
    cv = KFold(n_splits=splits, shuffle=True, random_state=42)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    scores = cross_val_score(pipeline, x_values, y_values, cv=cv, scoring="r2")
    return float(scores.mean())


def print_presentation_summary(
    baseline_results: dict,
    linear_results: dict,
    forest_results: dict,
    cross_val_mean: float | None,
) -> None:
    print("\n=== Key Metrics ===")
    print(f"Target: {TARGET_COLUMN}")
    print(
        f"Baseline Mean: R^2={baseline_results['r2']:.4f}, "
        f"MAE={baseline_results['mae']:.4f}"
    )
    print(
        f"Linear Regression: R^2={linear_results['r2']:.4f}, "
        f"MAE={linear_results['mae']:.4f}"
    )
    print(
        f"Random Forest: R^2={forest_results['r2']:.4f}, "
        f"MAE={forest_results['mae']:.4f}"
    )
    if cross_val_mean is not None:
        print(f"Cross-validation mean R^2: {cross_val_mean:.4f}")


def save_model(pipeline: Pipeline, model_name: str, results: dict, numeric_columns: list[str], boolean_columns: list[str], categorical_columns: list[str]) -> None:
    """Save the trained model and metadata."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    try:
        joblib.dump(pipeline, MODEL_PATH)
        logging.info(f"Model saved to {MODEL_PATH}")
        print(f"\nModel saved to {MODEL_PATH}")
        
        metadata = {
            "model_name": model_name,
            "training_date": datetime.now().isoformat(),
            "target_column": TARGET_COLUMN,
            "r2_score": float(results["r2"]),
            "mae": float(results["mae"]),
            "numeric_features": numeric_columns,
            "boolean_features": boolean_columns,
            "categorical_features": categorical_columns,
        }
        
        with open(MODEL_METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Metadata saved to {MODEL_METADATA_PATH}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")
        print(f"Error saving model: {e}")


def load_trained_model() -> tuple[Pipeline, dict] | None:
    """Load a previously trained model and metadata."""
    try:
        if not os.path.exists(MODEL_PATH):
            logging.warning(f"Model not found at {MODEL_PATH}")
            return None
        
        pipeline = joblib.load(MODEL_PATH)
        with open(MODEL_METADATA_PATH, "r") as f:
            metadata = json.load(f)
        
        logging.info(f"Loaded model from {MODEL_PATH}")
        return pipeline, metadata
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None


def print_conclusion(
    linear_results: dict,
    forest_results: dict,
    cross_val_mean: float | None,
) -> str:
    print("\n=== Conclusion ===")
    print(
        "The price_rating model provides defensible analytics results for class and app integration."
    )
    if forest_results["r2"] >= linear_results["r2"]:
        best_model = "Random Forest"
        print(
            f"Random Forest is the best model in this run with R^2={forest_results['r2']:.4f}."
        )
    else:
        best_model = "Linear Regression"
        print(
            f"Linear Regression is the best model in this run with R^2={linear_results['r2']:.4f}."
        )
    if cross_val_mean is not None:
        print(
            f"The cross-validation mean R^2 is {cross_val_mean:.4f}, "
            "which suggests the model captures a real but moderate signal."
        )
    return best_model


def train_regression(df: pd.DataFrame) -> None:
    numeric_columns, boolean_columns, categorical_columns = choose_feature_columns(df)
    feature_columns = numeric_columns + boolean_columns + categorical_columns
    if not feature_columns:
        raise SystemExit("No feature columns matched the current dataset.")

    x_values = df[feature_columns].copy()
    y_values = df[TARGET_COLUMN].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x_values, y_values, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        boolean_columns=boolean_columns,
        categorical_columns=categorical_columns,
    )

    baseline_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )
    linear_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(n_estimators=300, random_state=42),
            ),
        ]
    )

    baseline_results, _ = evaluate_model(
        "Baseline Mean", baseline_pipeline, x_train, x_test, y_train, y_test
    )
    linear_results, fitted_linear = evaluate_model(
        "Linear Regression", linear_pipeline, x_train, x_test, y_train, y_test
    )
    forest_results, fitted_forest = evaluate_model(
        "Random Forest", forest_pipeline, x_train, x_test, y_train, y_test
    )

    cross_val_mean = print_cross_validation(x_values, y_values, preprocessor)
    print_presentation_summary(
        baseline_results=baseline_results,
        linear_results=linear_results,
        forest_results=forest_results,
        cross_val_mean=cross_val_mean,
    )

    print_linear_coefficients(
        fitted_linear,
        numeric_columns=numeric_columns,
        boolean_columns=boolean_columns,
        categorical_columns=categorical_columns,
    )
    print_random_forest_importance(
        fitted_forest,
        numeric_columns=numeric_columns,
        boolean_columns=boolean_columns,
        categorical_columns=categorical_columns,
    )
    best_model = print_conclusion(
        linear_results=linear_results,
        forest_results=forest_results,
        cross_val_mean=cross_val_mean,
    )
    
    # Save the best model
    if best_model == "Random Forest":
        save_model(
            fitted_forest,
            "Random Forest",
            forest_results,
            numeric_columns,
            boolean_columns,
            categorical_columns,
        )
    else:
        save_model(
            fitted_linear,
            "Linear Regression",
            linear_results,
            numeric_columns,
            boolean_columns,
            categorical_columns,
        )


def main() -> None:
    logging.info("Starting plant regression training")
    args = parse_args()
    raw_dataframe, source_name = load_dataframe(args)
    prepared_dataframe = add_engineered_features(raw_dataframe)
    validated_dataframe = validate_dataframe(prepared_dataframe)
    numeric_columns, boolean_columns, categorical_columns = choose_feature_columns(
        validated_dataframe
    )
    print_dataset_summary(
        source_name=source_name,
        raw_df=raw_dataframe,
        model_df=validated_dataframe,
        numeric_columns=numeric_columns,
        boolean_columns=boolean_columns,
        categorical_columns=categorical_columns,
    )
    train_regression(validated_dataframe)
    logging.info("Training complete")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nCancelled.")
