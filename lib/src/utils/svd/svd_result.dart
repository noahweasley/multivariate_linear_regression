import 'dart:math';

import 'package:multivariate_linear_regression/src/utils/constants.dart';
import 'package:multivariate_linear_regression/src/utils/svd/matrix.dart';

/// Holds the result of a Singular Value Decomposition (SVD).
class SVDResult {
  /// Left singular vectors (m × n)
  final Matrix leftSingleVectors;

  /// Right singular vectors (n × n)
  final Matrix rightSingularVectors;

  /// Singular values (length = n)
  final List<double> singularValues;

  /// row count
  final int rows;

  /// column count
  final int cols;

  ///
  SVDResult({
    required this.leftSingleVectors,
    required this.rightSingularVectors,
    required this.singularValues,
    required this.rows,
    required this.cols,
  });

  /// condition number
  double get condition {
    return singularValues[0] / singularValues[min(rows, cols) - 1];
  }

  /// 2-norm (largest singular value)
  double get norm2 {
    return singularValues[0];
  }

  /// numerical rank
  int get rank {
    final tol = max(rows, cols) * singularValues[0] * epsilon;
    var r = 0;

    for (final v in singularValues) {
      if (v > tol) {
        r++;
      }
    }

    return r;
  }

  /// diagonal values
  List<double> get diagonal {
    return singularValues;
  }

  /// threshold
  double get threshold {
    return (epsilon / 2) * max(rows, cols) * singularValues[0];
  }

  /// inverse of svd results
  Matrix inverse() {
    final V = rightSingularVectors;
    final e = threshold;
    final s = singularValues;

    final vrows = V.rows;
    final Vcols = V.cols;

    final X = Matrix.zeros(vrows, s.length);

    for (var i = 0; i < vrows; i++) {
      for (var j = 0; j < Vcols; j++) {
        if (s[j].abs() > e) {
          X.set(i, j, V.get(i, j) / s[j]);
        }
      }
    }

    final U = leftSingleVectors;
    final Urows = U.rows;
    final Ucols = U.cols;

    final Y = Matrix.zeros(vrows, Urows);

    for (var i = 0; i < vrows; i++) {
      for (var j = 0; j < Urows; j++) {
        var sum = 0.0;

        for (var k = 0; k < Ucols; k++) {
          sum += X.get(i, k) * U.get(j, k);
        }

        Y.set(i, j, sum);
      }
    }

    return Y;
  }

  /// Σ matrix
  Matrix get diagonalMatrix {
    return Matrix.diag(singularValues);
  }
}
