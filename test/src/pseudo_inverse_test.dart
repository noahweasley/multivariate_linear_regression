import 'package:multivariate_linear_regression/src/svd/matrix.dart';
import 'package:multivariate_linear_regression/src/svd/pseudo_inverse.dart';
import 'package:test/test.dart';

void main() {
  group('Matrix pseudoInverse (SVD-based)', () {
    test('identity matrix returns itself', () {
      final I = Matrix.identity(3);
      final pInv = I.pseudoInverse();

      expect(pInv.toList(), equals(I.toList()));
    });

    test('tall rectangular matrix (3x2)', () {
      final A = Matrix.fromList([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);

      final aInverse = A.pseudoInverse();

      expect(aInverse.rows, equals(2));
      expect(aInverse.cols, equals(3));

      // Moore–Penrose
      final reconstructed = aInverse.multiply(A).multiply(aInverse);

      for (var i = 0; i < aInverse.rows; i++) {
        for (var j = 0; j < aInverse.cols; j++) {
          expect(
            (reconstructed.get(i, j) - aInverse.get(i, j)).abs() < 1e-10,
            isTrue,
          );
        }
      }
    });

    // TODO(noah): Fix test case
    test('empty matrix returns transpose', () {
      final A = Matrix.zeros(0, 0);
      final aInverse = A.pseudoInverse();

      expect(aInverse.rows, equals(0));
      expect(aInverse.cols, equals(0));
    });

    test('rank-deficient matrix zeroes small singular values', () {
      // Second column is a multiple of the first, rank 1
      final A = Matrix.fromList([
        [1, 2],
        [2, 4],
        [3, 6],
      ]);

      final aInverse = A.pseudoInverse(threshold: 1e-8);

      // Moore–Penrose
      final reconstructed = aInverse.multiply(A).multiply(aInverse);

      for (var i = 0; i < aInverse.rows; i++) {
        for (var j = 0; j < aInverse.cols; j++) {
          expect(
            (reconstructed.get(i, j) - aInverse.get(i, j)).abs() < 1e-8,
            isTrue,
          );
        }
      }
    });
  });
}
