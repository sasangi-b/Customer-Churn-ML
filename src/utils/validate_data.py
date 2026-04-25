import great_expectations as ge
from typing import Tuple, List
import pandas as pd


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.

    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    """
    print("🔍 Starting data validation with Great Expectations...")

    # ── Bootstrap GE context (ephemeral = no disk writes / no project needed) ──
    context = ge.get_context()

    datasource = context.sources.add_pandas("telco_source")
    data_asset = datasource.add_dataframe_asset("telco_asset")

    # Pre-process TotalCharges before handing the frame to GE
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    batch_request = data_asset.build_batch_request(dataframe=df)

    suite_name = "telco_suite"
    context.add_or_update_expectation_suite(suite_name)

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name,
    )

    # ── Helper: run an expectation and collect failures ───────────────────────
    failures: List[str] = []

    def check(result, message: str) -> None:
        if not result.success:
            failures.append(message)

    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    print("   📋 Validating schema and required columns...")

    # Customer identifier must exist (required for business operations)
    check(validator.expect_column_to_exist("customerID"),
          "Missing required column: 'customerID'")
    check(validator.expect_column_values_to_not_be_null("customerID"),
          "Null values found in 'customerID'")

    # Core demographic features
    for col in ("gender", "Partner", "Dependents"):
        check(validator.expect_column_to_exist(col),
              f"Missing required column: '{col}'")

    # Service features (critical for churn analysis)
    for col in ("PhoneService", "InternetService", "Contract"):
        check(validator.expect_column_to_exist(col),
              f"Missing required column: '{col}'")

    # Financial features (key churn predictors)
    for col in ("tenure", "MonthlyCharges", "TotalCharges"):
        check(validator.expect_column_to_exist(col),
              f"Missing required column: '{col}'")

    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")

    # Gender must be one of expected values (data integrity)
    check(validator.expect_column_values_to_be_in_set("gender", ["Male", "Female"]),
          "Invalid values found in 'gender'")

    # Yes/No fields must have valid values
    for col in ("Partner", "Dependents", "PhoneService"):
        check(validator.expect_column_values_to_be_in_set(col, ["Yes", "No"]),
              f"Invalid Yes/No values found in '{col}'")

    # Contract types must be valid (business constraint)
    check(
        validator.expect_column_values_to_be_in_set(
            "Contract", ["Month-to-month", "One year", "Two year"]
        ),
        "Invalid contract type values found in 'Contract'",
    )

    # Internet service types (business constraint)
    check(
        validator.expect_column_values_to_be_in_set(
            "InternetService", ["DSL", "Fiber optic", "No"]
        ),
        "Invalid internet service values found in 'InternetService'",
    )

    # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges and business constraints...")

    # Tenure must be non-negative (business logic - can't have negative tenure)
    check(
        validator.expect_column_values_to_be_between("tenure", min_value=0),
        "'tenure' contains negative values",
    )

    # Monthly charges must be positive (business logic - no free service)
    check(
        validator.expect_column_values_to_be_between("MonthlyCharges", min_value=0),
        "'MonthlyCharges' contains negative values",
    )

    # Total charges should be non-negative (business logic)
    check(
        validator.expect_column_values_to_be_between("TotalCharges", min_value=0),
        "'TotalCharges' contains negative values",
    )

    # === STATISTICAL VALIDATION ===
    print("   📈 Validating statistical properties...")

    # Tenure should be reasonable (max ~10 years = 120 months for telecom)
    check(
        validator.expect_column_values_to_be_between(
            "tenure", min_value=0, max_value=120
        ),
        "'tenure' values outside expected range [0, 120]",
    )

    # Monthly charges should be within reasonable business range
    check(
        validator.expect_column_values_to_be_between(
            "MonthlyCharges", min_value=0, max_value=200
        ),
        "'MonthlyCharges' values outside expected range [0, 200]",
    )

    # No missing values in critical numeric features
    check(
        validator.expect_column_values_to_not_be_null("tenure"),
        "Null values found in 'tenure'",
    )
    check(
        validator.expect_column_values_to_not_be_null("MonthlyCharges"),
        "Null values found in 'MonthlyCharges'",
    )

    # === DATA CONSISTENCY CHECKS ===
    print("   🔗 Validating data consistency...")

    # Total charges should generally be >= Monthly charges (except for very new customers)
    # This is a business logic check to catch data entry errors
    check(
        validator.expect_column_pair_values_A_to_be_greater_than_B(
            column_A="TotalCharges",
            column_B="MonthlyCharges",
            or_equal=True,
            mostly=0.95,  # Allow 5% exceptions for edge cases
        ),
        "TotalCharges is less than MonthlyCharges in more than 5% of rows",
    )

    # === RUN VALIDATION SUITE ===
    print("   ⚙️  Running complete validation suite...")
    results = validator.validate()

    # === PROCESS RESULTS ===
    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks

    # Collect expectation-type names for any suite-level failures not already captured
    failed_expectation_types = [
        r["expectation_config"]["expectation_type"]
        for r in results["results"]
        if not r["success"]
    ]

    overall_success = len(failures) == 0

    if overall_success:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectation_types}")

    return overall_success, failures