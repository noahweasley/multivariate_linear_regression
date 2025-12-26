import 'package:multivariate_linear_regression/src/svd/matrix.dart';
import 'package:multivariate_linear_regression/src/svd/pseudo_inverse.dart';
import 'package:test/test.dart';

void main() {
  group('Matrix pseudoInverse', () {
    test('identity matrix', () {
      final I = Matrix.identity(3);
      final IInverse = I.pseudoInverse();

      expect(IInverse.toList(), equals(I.toList()));
    });

    //  test('rectangular 2x3 matrix', () {
    //    final A = Matrix.fromList([
    //      [1, 2, 3],
    //      [4, 5, 6]
    //    ]);

    //    final AInverse = A.pseudoInverse();

    //    expect(AInverse.rows, equals(A.cols));
    //    expect(AInverse.cols, equals(A.rows));

    //    final reconstructed = A.multiply(AInverse).multiply(A);

    //    for (var i = 0; i < A.rows; i++) {
    //      for (var j = 0; j < A.cols; j++) {
    //        expect((reconstructed.get(i, j) - A.get(i, j)).abs() < 1e-10, isTrue);
    //      }
    //    }
    //  });

    test('rectangular 3x2 matrix', () {
      final A = Matrix.fromList([
        [1, 2],
        [3, 4],
        [5, 6]
      ]);

      final AInverse = A.pseudoInverse();

      expect(AInverse.rows, equals(A.cols));
      expect(AInverse.cols, equals(A.rows));

      final reconstructed = AInverse.multiply(A).multiply(AInverse);
      for (var i = 0; i < AInverse.rows; i++) {
        for (var j = 0; j < AInverse.cols; j++) {
          expect((reconstructed.get(i, j) - AInverse.get(i, j)).abs() < 1e-10, isTrue);
        }
      }
    });

    //  test('empty matrix returns transpose', () {
    //    final A = Matrix.zeros(0, 3);
    //    final AInverse = A.pseudoInverse();

    //    expect(AInverse.rows, equals(3));
    //    expect(AInverse.cols, equals(0));
    //  });
  });
}
