// ignore_for_file: public_member_api_docs
import 'dart:math';
import 'package:multivariate_linear_regression/src/utils/svd/matrix.dart';
import 'package:multivariate_linear_regression/src/utils/svd/pseudo_inverse.dart';
import 'package:multivariate_linear_regression/src/utils/svd/svd.dart';

/// Author: Ebenmelu Ifechukwu (@noahweasley)
///
/// Multivariate linear regression implemented using Golub-Reinsch Singular Value Decomposition
/// (SVD) to improve numerical stability.
///
/// The regression solves the normal equation:
///
///   beta = inverse(transpose(X) * X) * transpose(X) * Y
///
/// where:
/// - X is the input (design) matrix
/// - Y is the output matrix
/// - beta is the coefficient matrix
class MultivariateLinearRegression {
  /// Creates a multivariate linear regression model.
  ///
  /// x: input matrix with shape (rows x features)
  /// y: output matrix with shape (rows x outputs)
  ///
  /// If intercept is true, a column of ones is appended to X.
  /// If statistics is true, variance and inference statistics
  /// are computed.
  MultivariateLinearRegression({
    required List<List<double>> x,
    required List<List<double>> y,
    this.intercept = true,
    this.statistics = true,
  }) {
    _originalX = x;
    _originalY = y;

    _x = Matrix.fromList(x);
    _y = Matrix.fromList(y);

    if (intercept) {
      final ones = Matrix.fromList(
        List.generate(_x.rows, (_) => [1.0]),
      );
      _x = _x.appendColumn(ones);
    }

    _inputs = _x.cols - (intercept ? 1 : 0);
    _outputs = _y.cols;

    _beta = _computeBeta();

    if (statistics) {
      _variance = _computeVariance();
    }
  }

  /// Recreates a model using the original training data.
  factory MultivariateLinearRegression.load(
    MultivariateLinearRegression model,
  ) {
    return MultivariateLinearRegression(
      x: model._originalX,
      y: model._originalY,
      intercept: model.intercept,
      statistics: model.statistics,
    );
  }

  /// Design matrix X
  late Matrix _x;

  /// Output matrix Y
  late Matrix _y;

  /// Coefficient matrix beta
  late Matrix _beta;

  /// Number of input features
  late int _inputs;

  /// Number of output variables
  late int _outputs;

  /// Residual variance
  double? _variance;

  /// Original input data
  late List<List<double>> _originalX;

  /// Original output data
  late List<List<double>> _originalY;

  /// Whether an intercept column is used
  final bool intercept;

  /// Whether statistics are computed
  final bool statistics;

  /// Number of input features
  int get inputs => _inputs;

  /// Number of output variables
  int get outputs => _outputs;

  /// Regression coefficients
  List<List<double>> get weights => _beta.toList();

  /// Standard error of the regression
  double? get stdError => _variance == null ? null : sqrt(_variance!);

  /// Covariance matrix of coefficients
  ///
  /// cov(beta) = variance * inverse(transpose(X) * X)
  ///
  /// Throws StateError if statistics are disabled.
  Matrix get stdErrorMatrix {
    if (_variance == null) {
      throw StateError('Statistics disabled');
    }

    final xtxInv = _x.transpose().multiply(_x).pseudoInverse();
    return xtxInv.scale(_variance!);
  }

  /// Standard error for each coefficient
  List<double> get stdErrors => stdErrorMatrix.diagonal().map(sqrt).toList();

  /// t-statistics for each coefficient
  ///
  /// t = beta / standard_error
  List<double> get tStats {
    final errors = stdErrors;
    final coefficient = weights;

    return List.generate(coefficient.length, (i) {
      final beta = coefficient[i][0];
      return errors[i] == 0 ? 0.0 : beta / errors[i];
    });
  }

  /// Computes regression coefficients.
  ///
  /// Steps:
  /// 1. Xt  = transpose(X)
  /// 2. XtX = Xt * X
  /// 3. XtY = Xt * Y
  /// 4. beta = inverse(XtX) * XtY
  ///
  /// The inverse is computed using SVD.
  Matrix _computeBeta() {
    final xt = _x.transpose();
    final xx = xt.multiply(_x);
    final xy = xt.multiply(_y);

    final svdResults = GolubReinschSVD.decompose(xx).results;
    final invxx = svdResults.inverse();

    return xy.transpose().multiply(invxx).transpose();
  }

  /// Computes residual variance.
  ///
  /// residual = actual - predicted
  ///
  /// variance =
  ///   sum(residual * residual) /
  ///   (number_of_rows - number_of_columns)
  double _computeVariance() {
    final fitted = _x.multiply(_beta);
    final residuals = _y.clone().add(fitted.neg());

    return residuals.toList().fold(
              0.0,
              (result, residual) => result + pow(residual[0], 2),
            ) /
        (_y.rows - _x.cols);
  }

  /// Predicts output values for a single input vector.
  ///
  /// Throws ArgumentError if input size is incorrect.
  List<double> predict(List<double> x) {
    if (x.length != inputs) {
      throw ArgumentError(
        'Expected $inputs inputs, got ${x.length}',
      );
    }

    final result = List<double>.filled(outputs, 0.0);

    if (intercept) {
      for (var j = 0; j < outputs; j++) {
        result[j] = _beta.get(inputs, j);
      }
    }

    for (var i = 0; i < inputs; i++) {
      for (var j = 0; j < outputs; j++) {
        result[j] += _beta.get(i, j) * x[i];
      }
    }

    return result;
  }

  /// Predicts outputs for multiple input rows.
  List<List<double>> predictBatch(
    List<List<double>> x,
  ) =>
      x.map(predict).toList();

  /// Converts the model to a JSON-compatible map.
  Map<String, dynamic> toJson() {
    return {
      'name': 'multivariateLinearRegression',
      'weights': weights,
      'inputs': inputs,
      'outputs': outputs,
      'intercept': intercept,
      'summary': statistics
          ? {
              'regressionStatistics': {
                'standardError': stdError,
              },
              'variables': List.generate(
                weights.length,
                (i) {
                  return {
                    'label': i == weights.length - 1 ? 'Intercept' : 'X${i + 1}',
                    'coefficients': weights[i],
                    'standardError': stdErrors[i],
                    'tStat': tStats[i],
                  };
                },
              ),
            }
          : null,
    };
  }
}
