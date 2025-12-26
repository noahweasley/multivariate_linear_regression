import 'package:multivariate_linear_regression/src/svd/matrix.dart';
import 'package:test/test.dart';

void main() {
  group('Matrix', () {
    test('fromList and clone', () {
      final A = Matrix.fromList([
        [1, 2],
        [3, 4]
      ]);

      expect(A.rows, equals(2));
      expect(A.cols, equals(2));
      expect(A.get(0, 0), equals(1));
      expect(A.get(1, 1), equals(4));

      final B = A.clone();
      expect(B.toList(), equals(A.toList()));

      B.set(0, 0, 100);
      expect(A.get(0, 0), equals(1), reason: 'Clone should be deep copy');
    });

    test('zeros and identity', () {
      final Z = Matrix.zeros(2, 3);
      final I = Matrix.identity(3);

      expect(Z.rows, equals(2));
      expect(Z.cols, equals(3));

      expect(
        Z.toList(),
        equals([
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0]
        ]),
      );

      expect(
          I.toList(),
          equals([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
          ]));
    });

    test('transpose', () {
      final A = Matrix.fromList([
        [1, 2, 3],
        [4, 5, 6]
      ]);

      final T = A.transpose();

      expect(
        T.toList(),
        equals([
          [1, 4],
          [2, 5],
          [3, 6]
        ]),
      );
    });

    test('multiply', () {
      final A = Matrix.fromList([
        [1, 2],
        [3, 4]
      ]);

      final B = Matrix.fromList([
        [5, 6],
        [7, 8]
      ]);

      final C = A.multiply(B);

      expect(
        C.toList(),
        equals([
          [19, 22],
          [43, 50]
        ]),
      );
    });

    test('add, subtract, neg, scale', () {
      final A = Matrix.fromList([
        [1, 2],
        [3, 4]
      ]);

      final B = Matrix.fromList([
        [5, 6],
        [7, 8]
      ]);

      expect(
        A.add(B).toList(),
        equals([
          [6, 8],
          [10, 12]
        ]),
      );

      expect(
          B.subtract(A).toList(),
          equals([
            [4, 4],
            [4, 4]
          ]));

      expect(
        A.neg().toList(),
        equals([
          [-1, -2],
          [-3, -4]
        ]),
      );

      expect(
          A.scale(2).toList(),
          equals([
            [2, 4],
            [6, 8]
          ]));
    });

    test('appendColumn', () {
      final A = Matrix.fromList([
        [1, 2],
        [3, 4]
      ]);

      final B = Matrix.fromList([
        [5],
        [6]
      ]);

      final C = A.appendColumn(B);

      expect(
          C.toList(),
          equals([
            [1, 2, 5],
            [3, 4, 6]
          ]));
    });

    test('diag and diagonal', () {
      final D = Matrix.diag([1, 2, 3]);

      expect(
          D.toList(),
          equals([
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]
          ]));

      expect(D.diagonal(), equals([1, 2, 3]));
    });

    test('set and get', () {
      final A = Matrix.zeros(2, 2);
      A.set(0, 1, 42);

      expect(A.get(0, 1), equals(42));
    });

    test('shape mismatch errors', () {
      final A = Matrix.zeros(2, 2);
      final B = Matrix.zeros(3, 2);
      final C = Matrix.zeros(2, 3);

      expect(() => A.add(B), throwsArgumentError);
      expect(() => A.subtract(B), throwsArgumentError);
      expect(() => A.multiply(C), returnsNormally);
      expect(() => A.appendColumn(Matrix.zeros(3, 1)), throwsArgumentError);
    });
  });
}
