// ignore_for_file: public_member_api_docs
import 'dart:math';
import 'package:multivariate_linear_regression/src/utils/svd/matrix.dart';
import 'package:multivariate_linear_regression/src/utils/svd/pseudo_inverse.dart';
import 'package:multivariate_linear_regression/src/utils/svd/svd.dart';

/// Author: Ebenmelu Ifechukwu @noahweasley
///
/// {@template multivariate_linear_regression}
/// Multivariate linear regression with optional intercept implemented using Golub-Reinsch Singular Value Decomposition.
/// {@endtemplate}
class MultivariateLinearRegression {
  /// {@macro multivariate_linear_regression}
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
      final ones = Matrix.fromList(List.generate(_x.rows, (_) => [1.0]));
      _x = _x.appendColumn(ones);
    }

    _inputs = _x.cols - (intercept ? 1 : 0);
    _outputs = _y.cols;

    _beta = _computeBeta();

    if (statistics) {
      _variance = _computeVariance();
    }
  }

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

  late Matrix _x;
  late Matrix _y;
  late Matrix _beta;

  late int _inputs;
  late int _outputs;

  double? _variance;

  late List<List<double>> _originalX;
  late List<List<double>> _originalY;

  final bool intercept;
  final bool statistics;

  int get inputs => _inputs;
  int get outputs => _outputs;

  List<List<double>> get weights => _beta.toList();

  double? get stdError => _variance == null ? null : sqrt(_variance!);

  Matrix get stdErrorMatrix {
    if (_variance == null) {
      throw StateError('Statistics disabled');
    }

    final xtxInv = _x.transpose().multiply(_x).pseudoInverse();
    return xtxInv.scale(_variance!);
  }

  List<double> get stdErrors => stdErrorMatrix.diagonal().map(sqrt).toList();

  List<double> get tStats {
    final errors = stdErrors;
    final coefficient = weights;

    return List.generate(coefficient.length, (i) {
      final beta = coefficient[i][0];
      return errors[i] == 0 ? 0.0 : beta / errors[i];
    });
  }

  Matrix _computeBeta() {
    final xt = _x.transpose();
    final xx = xt.multiply(_x);
    final xy = xt.multiply(_y);

    final svdResults = GolubReinschSVD.decompose(xx).results;
    final invxx = svdResults.inverse();

    return xy.transpose().multiply(invxx).transpose();
  }

  double _computeVariance() {
    final fitted = _x.multiply(_beta);
    final residuals = _y.clone().add(fitted.neg());

    return residuals.toList().fold(0.0, (result, residual) => result + pow(residual[0], 2)) / (_y.rows - _x.cols);
  }

  List<double> predict(List<double> x) {
    if (x.length != inputs) {
      throw ArgumentError('Expected $inputs inputs, got ${x.length}');
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

  List<List<double>> predictBatch(List<List<double>> x) => x.map(predict).toList();

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
              'variables': List.generate(weights.length, (i) {
                return {
                  'label': i == weights.length - 1 ? 'Intercept' : 'X${i + 1}',
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
