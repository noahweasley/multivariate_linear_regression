import 'dart:math';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:multivariate_linear_regression/src/utils/matrix_ext.dart';

/// {@template multivariate_linear_regression}
/// Multivariate linear regression with optional intercept.
/// {@endtemplate}
class MultivariateLinearRegression {
  /// {@macro multivariate_linear_regression}
  MultivariateLinearRegression({
    required List<List<double>> x,
    required List<List<double>> y,
    this.intercept = true,
    this.statistics = true,
  }) {
    _x = Matrix.fromList(x);
    _y = Matrix.fromList(y);

    if (intercept) {
      final ones = Vector.filled(_x.rowCount, 1);
      _x = _x.insertColumns(_x.columnCount, List.from(ones));
    }

    _inputs = _x.columnCount - (intercept ? 1 : 0);
    _outputs = _y.columnCount;

    _beta = _computeBeta();

    if (statistics) {
      _variance = _computeVariance();
    }
  }

  /// Load previous model
  factory MultivariateLinearRegression.load(MultivariateLinearRegression model) {
    return MultivariateLinearRegression(
      x: model._originalX,
      y: model._originalY,
      intercept: model.intercept,
      statistics: model.statistics,
    );
  }

  late final Matrix _x;
  late final Matrix _y;
  late final Matrix _beta;

  late final int _inputs;
  late final int _outputs;

  double? _variance;

  late final List<List<double>> _originalX = _x.toList().map(List<double>.from).toList();
  late final List<List<double>> _originalY = _y.toList().map(List<double>.from).toList();

  ///
  final bool intercept;

  ///
  final bool statistics;

  ///
  int get inputs => _inputs;

  ///
  int get outputs => _outputs;

  /// Regression coefficients (features × outputs)
  List<List<double>> get weights => _beta.toList().map(List<double>.from).toList();

  /// Standard error (sqrt of variance)
  double? get stdError => _variance == null ? null : sqrt(_variance!);

  /// (XᵀX)⁻¹ used for coefficient statistics
  Matrix get stdErrorMatrix => _x.transpose().multiply(_x).pseudoInverse();

  /// Standard errors for each coefficient
  List<double> get stdErrors {
    final varMatrix = stdErrorMatrix * (_variance ?? 0);
    return List.generate(varMatrix.rowCount, (i) => sqrt(varMatrix[i][i]));
  }

  /// t-statistics for each coefficient
  List<double> get tStats {
    final errors = stdErrors;
    return List.generate(weights.length, (i) {
      final coefficient = weights[i].first;
      return errors[i] == 0 ? 0.0 : coefficient / errors[i];
    });
  }

  Matrix _computeBeta() {
    return _x.pseudoInverse().multiply(_y);
  }

  double _computeVariance() {
    final fitted = _x.multiply(_beta);
    final residuals = _y - fitted;

    final squaredSum = residuals.toList().map((ri) => pow(ri.first, 2)).reduce((a, b) => a + b);

    return squaredSum / (_y.rowCount - _x.columnCount);
  }

  ///
  List<double> predict(List<double> x) {
    if (x.length != inputs) {
      throw ArgumentError('Expected $inputs inputs, got ${x.length}');
    }

    final row = intercept ? [...x, 1.0] : x;
    final result = List<double>.filled(outputs, 0);

    for (var j = 0; j < outputs; j++) {
      for (var i = 0; i < row.length; i++) {
        result[j] += _beta[i][j] * row[i];
      }
    }

    return result;
  }

  ///
  List<List<double>> predictBatch(List<List<double>> x) {
    return x.map(predict).toList();
  }

  ///
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
                'observations': outputs,
              },
              'variables': List.generate(weights.length, (i) {
                return {
                  'label': i == weights.length - 1 ? 'Intercept' : 'X Variable ${i + 1}',
                  'coefficients': weights[i],
                  'standardError': stdErrors[i],
                  'tStat': tStats[i],
                };
              }),
            }
          : null,
    };
  }
}
