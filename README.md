# Multivariate Linear Regression

[![Style: very good analysis][very_good_analysis_badge]][very_good_analysis_link]
![Test Coverage](https://img.shields.io/badge/Test%20coverage-96%25-green)
[![Powered by Mason](https://img.shields.io/endpoint?url=https%3A%2F%2Ftinyurl.com%2Fmason-badge)][mason_link]
[![License: MIT][license_badge]][license_link]

Multivariate linear regression for Dart with support for multiple outputs and optional intercept.

> **Inspired by [ml-matrix](https://github.com/mljs/matrix) and [regression-multivariate-linear](https://github.com/mljs/multivariate-linear-regression) Node.js libraries.**

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

- `MultivariateLinearRegression({x, y, intercept = true, statistics = true})`

  - Creates a regression model.
  - `x`: List of input rows.
  - `y`: List of output rows.
  - `intercept`: Whether to include an intercept column.
  - `statistics`: Whether to compute variance, standard errors, and t-stats.

- `MultivariateLinearRegression.load(model)`

  - Recreates a model using the original training data.

- `predict(List<double> x)`

  - Predicts outputs for a single input vector.

- `predictBatch(List<List<double>> x)`

  - Predicts outputs for multiple input rows.

- `weights`

  - Returns the regression coefficients.

- `stdError`

  - Standard error of the regression.

- `stdErrors`

  - Standard error for each coefficient.

- `tStats`

  - t-statistics for each coefficient.

- `stdErrorMatrix`

  - Covariance matrix of coefficients (requires `statistics = true`).

- `toJson()`

  - Converts the model to a JSON-compatible map, including regression statistics if enabled.

---

## Continuous Integration

Multivariate Linear Regression comes with a built-in [GitHub Actions workflow][github_actions_link] powered by [Very Good Workflows][very_good_workflows_link].

On each pull request and push, the CI formats, lints, and tests the code.
The project uses [Very Good Analysis][very_good_analysis_link] for a strict set of analysis rules.
Code coverage is enforced using [Very Good Coverage][very_good_coverage_link].

> **Note:** The coverage is currently at 96% due to a few small utility paths in SVD computations that are difficult to trigger via unit tests.

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

[dart_install_link]: https://dart.dev/get-dart
[github_actions_link]: https://docs.github.com/en/actions/learn-github-actions
[license_badge]: https://img.shields.io/badge/license-MIT-blue.svg
[license_link]: https://opensource.org/licenses/MIT
[mason_link]: https://github.com/felangel/mason
[very_good_analysis_badge]: https://img.shields.io/badge/style-very_good_analysis-B22C89.svg
[very_good_analysis_link]: https://pub.dev/packages/very_good_analysis
[very_good_coverage_link]: https://github.com/marketplace/actions/very-good-coverage
[very_good_workflows_link]: https://github.com/VeryGoodOpenSource/very_good_workflows
