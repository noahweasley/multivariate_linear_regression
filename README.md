# Multivariate Linear Regression

[![Style: very good analysis][very_good_analysis_badge]][very_good_analysis_link]
![Test Coverage](https://img.shields.io/badge/Test%20coverage-96.4%25-green)
[![Powered by Mason](https://img.shields.io/endpoint?url=https%3A%2F%2Ftinyurl.com%2Fmason-badge)][mason_link]
[![License: MIT][license_badge]][license_link]

Multivariate linear regression for Dart with support for multiple outputs and optional intercept, implemented using Golub-Reinsch Singular Value Decomposition.

> **Inspired by [ml-matrix](https://github.com/mljs/matrix) and [regression-multivariate-linear](https://github.com/mljs/regression-multivariate-linear) Node.js libraries.**

---

## Installation

**In order to start using Multivariate Linear Regression you must have the [Dart SDK][dart_install_link] installed on your machine.**

Install via `dart pub add`:

```sh
dart pub add multivariate_linear_regression
```

---

## Usage

```dart
import 'package:multivariate_linear_regression/multivariate_linear_regression.dart';

void main() {
  final x = [
    [0.0, 0.0],
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
  ];

  final y = [
    [0.0, 0.0, 0.0],
    [2.0, 4.0, 3.0],
    [4.0, 6.0, 5.0],
    [6.0, 8.0, 7.0],
  ];

  final mlr = MultivariateLinearRegression(
    x: x,
    y: y,
  );

  print(mlr.predict([3.0, 3.0]));
  print(mlr.predictBatch([[1.0, 2.0], [2.0, 3.0]]));
  print(mlr.weights);
  print(mlr.stdError);
  print(mlr.stdErrors);
  print(mlr.tStats);
  print(mlr.toJson());
}
```

---

## API Overview

### Constructor

```dart
MultivariateLinearRegression({
  required List<List<double>> x,
  required List<List<double>> y,
  bool intercept = true,
  bool statistics = true,
})
```

Creates a multivariate linear regression model.

- `x` - Input feature matrix (rows = samples, columns = features)
- `y` — Output matrix (rows = samples, columns = targets)
- `intercept` — Includes a bias (intercept) term when set to `true`
- `statistics` — Enables computation of additional metrics (standard errors, t-stats, etc.)

---

### Load Existing Model

```dart
factory MultivariateLinearRegression.load(MultivariateLinearRegression model)
```

Reconstructs a trained model from previously trained model

---

### Prediction

```dart
List<double> predict(List<double> input)
```

Returns predicted outputs for a single input vector.

```dart
List<List<double>> predictBatch(List<List<double>> inputs)
```

Returns predictions for multiple input rows.

---

### Coefficients & Metrics

```dart
List<List<double>> get weights
```

Matrix of regression coefficients (includes intercept if enabled).

```dart
double get stdError
```

Overall standard error of the model.

```dart
List<List<double>> get stdErrors
```

Standard error for each coefficient.

```dart
List<List<double>> get tStats
```

T-statistics corresponding to each coefficient.

```dart
List<List<double>> get stdErrorMatrix
```

Covariance matrix of the coefficients.

> Available only when `statistics = true`

---

### Serialization

```dart
Map<String, dynamic> toJson()
```

Serializes the model into a JSON-compatible format, including statistics when enabled.

---

## Continuous Integration

Multivariate Linear Regression comes with a built-in [GitHub Actions workflow][github_actions_link] powered by [Very Good Workflows][very_good_workflows_link].

On each pull request and push, the CI formats, lints, and tests the code.
The project uses [Very Good Analysis][very_good_analysis_link] for a strict set of analysis rules.
Code coverage is enforced using [Very Good Coverage][very_good_coverage_link].

---

## Running Tests

To run all unit tests and generate coverage:

```sh
dart pub global activate coverage 1.15.0
dart test --coverage=coverage
dart pub global run coverage:format_coverage --lcov --in=coverage --out=coverage/lcov.info
```

To view the coverage report using lcov:

```sh
genhtml coverage/lcov.info -o coverage/
open coverage/index.html
```

---

## Support

If you find this package useful, please consider supporting it:

- Like the [package on pub.dev](https://pub.dev/packages/multivariate_linear_regression)
- Star the [GitHub repository](https://github.com/noahweasley/Multivariate-Linear-Regression)

Your support helps improve the project and keeps it actively maintained 😊

[dart_install_link]: https://dart.dev/get-dart
[github_actions_link]: https://docs.github.com/en/actions/learn-github-actions
[license_badge]: https://img.shields.io/badge/license-MIT-blue.svg
[license_link]: https://opensource.org/licenses/MIT
[mason_link]: https://github.com/felangel/mason
[very_good_analysis_badge]: https://img.shields.io/badge/style-very_good_analysis-B22C89.svg
[very_good_analysis_link]: https://pub.dev/packages/very_good_analysis
[very_good_coverage_link]: https://github.com/marketplace/actions/very-good-coverage
[very_good_workflows_link]: https://github.com/VeryGoodOpenSource/very_good_workflows
