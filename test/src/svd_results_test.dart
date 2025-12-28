import 'package:multivariate_linear_regression/src/svd/matrix.dart';
import 'package:multivariate_linear_regression/src/svd/svd.dart';
import 'package:test/test.dart';

void main() {
  group('Singular Value Decomposition results', () {
    test('basic SVD reconstruction A ≈ U Σ Vᵀ', () {
      final A = Matrix.fromList([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
      ]);

      final svd = GolubReinschSVD.decompose(A).results;

      final U = svd.leftSingleVectors;
      final S = svd.diagonalMatrix;
      final V = svd.rightSingularVectors;

      final reconstructed = U.multiply(S).multiply(V.transpose());

      const tol = 1e-9;

      for (var i = 0; i < A.rows; i++) {
        for (var j = 0; j < A.cols; j++) {
          expect(
            reconstructed.get(i, j),
            closeTo(A.get(i, j), tol),
          );
        }
      }

      expect(svd.norm2, greaterThan(0));
      expect(svd.rank, equals(2));
      expect(svd.condition, greaterThanOrEqualTo(1));
    });

    test('singular values are sorted descending', () {
      final A = Matrix.fromList([
        [3.0, 1.0],
        [1.0, 3.0],
      ]);

      final s = GolubReinschSVD.decompose(A).results.singularValues;

      for (var i = 0; i < s.length - 1; i++) {
        expect(s[i], greaterThanOrEqualTo(s[i + 1]));
      }
    });

    test('rank-deficient matrix has reduced rank', () {
      final A = Matrix.fromList([
        [1.0, 2.0],
        [2.0, 4.0],
      ]);

      final svd = GolubReinschSVD.decompose(A).results;

      expect(svd.rank, equals(1));
      expect(svd.singularValues.last, closeTo(0.0, 1e-12));
    });

    test('inverse via SVD produces valid pseudoinverse', () {
      final A = Matrix.fromList([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);

      final svd = GolubReinschSVD.decompose(A).results;
      final svdInverse = svd.inverse();

      // A * A⁺ ≈ I
      final identity = A.multiply(svdInverse);

      const tol = 1e-9;

      for (var i = 0; i < identity.rows; i++) {
        for (var j = 0; j < identity.cols; j++) {
          if (i == j) {
            expect(identity.get(i, j), closeTo(1.0, tol));
          } else {
            expect(identity.get(i, j), closeTo(0.0, tol));
          }
        }
      }
    });

    test('zero matrix has zero rank and zero norm', () {
      final A = Matrix.zeros(3, 2);

      final svd = GolubReinschSVD.decompose(A).results;

      expect(svd.rank, equals(0));
      expect(svd.norm2, equals(0));
    });

    test('diagonal getter and condition are consistent', () {
      final A = Matrix.fromList([
        [4.0, 0.0],
        [0.0, 2.0],
      ]);

      final svd = GolubReinschSVD.decompose(A).results;
      final d = svd.diagonal;

      expect(d.length, equals(2));
      expect(
        svd.condition,
        closeTo(d.first / d.last, 1e-12),
      );
    });

    test('wide matrix (more columns than rows)', () {
      final A = Matrix.fromList([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);

      final svd = GolubReinschSVD.decompose(A).results;

      expect(svd.rows, equals(2));
      expect(svd.cols, equals(3));
      expect(svd.rank, equals(2));
    });

    test('tall matrix (more rows than columns)', () {
      final A = Matrix.fromList([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
      ]);

      final svd = GolubReinschSVD.decompose(A).results;

      expect(svd.rows, equals(4));
      expect(svd.cols, equals(2));
      expect(svd.rank, equals(2));
    });
  });
}
