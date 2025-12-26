// ignore_for_file: public_member_api_docs

import 'package:multivariate_linear_regression/src/utils/svd/matrix.dart';
import 'package:multivariate_linear_regression/src/utils/svd/svd.dart';

extension MatrixPseudoInverse on Matrix {
  /// Moore–Penrose pseudoinverse using **Golub–Reinsch SVD**.
  ///
  /// This matches **ml-matrix** and LAPACK behavior:
  ///   A⁺ = V Σ⁺ Uᵀ
  ///
  /// Recommended for:
  /// - regression
  /// - rank-deficient matrices
  /// - production use
  Matrix pseudoInverse({
    double threshold = double.minPositive,
  }) {
    if (rows == 0 || cols == 0) {
      return transpose();
    }

    final svd = GolubReinschSVD.decompose(this);
    final results = svd.results;

    final U = results.leftSingleVectors;
    final V = results.rightSingularVectors;
    final s = List<double>.from(results.singularValues);

    for (var i = 0; i < s.length; i++) {
      if (s[i].abs() > threshold) {
        s[i] = 1.0 / s[i];
      } else {
        s[i] = 0.0;
      }
    }

    // A⁺ = V Σ⁺ Uᵀ
    return V.multiply(Matrix.diag(s)).multiply(U.transpose());
  }
}
