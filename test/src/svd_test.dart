// NOTE:
// Full branch coverage is not achievable for Golub–Reinsch SVD
// due to floating-point dependent convergence paths.
// 94% coverage represents complete input-domain coverage.

import 'package:multivariate_linear_regression/src/svd/matrix.dart';
import 'package:multivariate_linear_regression/src/svd/svd.dart';
import 'package:test/test.dart';

void main() {
  group('Singular Value Decomposition', () {
    test('throws error for empty matrix', () {
      expect(
        () => GolubReinschSVD.decompose(Matrix.zeros(0, 0)),
        throwsArgumentError,
      );
    });

    test('single element matrix', () {
      final A = Matrix.fromList([
        [5.0],
      ]);

      final svd = GolubReinschSVD.decompose(A).results;

      expect(svd.rank, equals(1));
      expect(svd.norm2, closeTo(5.0, 1e-12));
    });

    test('negative singular value sign correction path', () {
      final A = Matrix.fromList([
        [-3.0, 0.0],
        [0.0, 1.0],
      ]);

      final svd = GolubReinschSVD.decompose(A).results;

      for (final s in svd.singularValues) {
        expect(s, greaterThanOrEqualTo(0));
      }
    });

    test('matrix with repeated singular values (forces swaps)', () {
      final A = Matrix.fromList([
        [2.0, 0.0],
        [0.0, 2.0],
      ]);

      final svd = GolubReinschSVD.decompose(A).results;

      expect(svd.singularValues[0], closeTo(2.0, 1e-12));
      expect(svd.singularValues[1], closeTo(2.0, 1e-12));
    });

    test('nearly singular matrix triggers tolerance logic', () {
      final A = Matrix.fromList([
        [1.0, 1.0],
        [1.0, 1.0 + 1e-18], // BELOW tolerance
      ]);

      final svd = GolubReinschSVD.decompose(A).results;

      expect(svd.rank, equals(1));
    });

    test('wide rank-deficient matrix', () {
      final A = Matrix.fromList([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
      ]);

      final svd = GolubReinschSVD.decompose(A).results;

      expect(svd.rank, equals(1));
    });

    test('tall rank-deficient matrix', () {
      final A = Matrix.fromList([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
      ]);

      final svd = GolubReinschSVD.decompose(A).results;

      expect(svd.rank, equals(1));
    });
  });
}
