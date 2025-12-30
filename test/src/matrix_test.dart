import 'package:multivariate_linear_regression/src/svd/matrix.dart';
import 'package:test/test.dart';

void main() {
  group('Matrix', () {
    test('fromList and clone (deep copy)', () {
      final A = Matrix.fromList([
        [1, 2],
        [3, 4],
      ]);

      expect(A.rows, 2);
      expect(A.cols, 2);
      expect(A.get(0, 0), 1);
      expect(A.get(1, 1), 4);

      final B = A.clone();
      expect(B.toList(), equals(A.toList()));

      B.set(0, 0, 99);
      expect(A.get(0, 0), 1, reason: 'Clone must be deep copy');
    });

    test('fromList with empty list', () {
      final A = Matrix.fromList([]);
      expect(A.rows, 0);
      expect(A.cols, 0);
      expect(A.toList(), equals([]));
    });

    test('fromList throws on jagged matrix', () {
      expect(
        () => Matrix.fromList([
          [1, 2],
          [3],
        ]),
        throwsArgumentError,
      );
    });

    test('zeros and identity', () {
      final Z = Matrix.zeros(2, 3);
      final I = Matrix.identity(3);

      expect(Z.rows, 2);
      expect(Z.cols, 3);
      expect(
        Z.toList(),
        equals([
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
        ]),
      );

      expect(
        I.toList(),
        equals([
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0],
        ]),
      );
    });

    test('get and set', () {
      final A = Matrix.zeros(2, 2)..set(0, 1, 42);
      expect(A.get(0, 1), 42);
    });

    test('transpose square and rectangular', () {
      final A = Matrix.fromList([
        [1, 2, 3],
        [4, 5, 6],
      ]);

      final T = A.transpose();

      expect(
        T.toList(),
        equals([
          [1, 4],
          [2, 5],
          [3, 6],
        ]),
      );
    });

    test('transpose empty matrix', () {
      final A = Matrix.fromList([]);
      final T = A.transpose();

      expect(T.rows, 0);
      expect(T.cols, 0);
      expect(T.toList(), equals([]));
    });

    test('multiply', () {
      final A = Matrix.fromList([
        [1, 2],
        [3, 4],
      ]);

      final B = Matrix.fromList([
        [5, 6],
        [7, 8],
      ]);

      final C = A.multiply(B);

      expect(
        C.toList(),
        equals([
          [19, 22],
          [43, 50],
        ]),
      );
    });

    test('multiply throws on shape mismatch', () {
      final A = Matrix.zeros(2, 3);
      final B = Matrix.zeros(4, 2);

      expect(() => A.multiply(B), throwsArgumentError);
    });

    test('add, subtract, neg, scale', () {
      final A = Matrix.fromList([
        [1, 2],
        [3, 4],
      ]);

      final B = Matrix.fromList([
        [5, 6],
        [7, 8],
      ]);

      expect(
        A.add(B).toList(),
        equals([
          [6, 8],
          [10, 12],
        ]),
      );

      expect(
        B.subtract(A).toList(),
        equals([
          [4, 4],
          [4, 4],
        ]),
      );

      expect(
        A.neg().toList(),
        equals([
          [-1, -2],
          [-3, -4],
        ]),
      );

      expect(
        A.scale(2).toList(),
        equals([
          [2, 4],
          [6, 8],
        ]),
      );
    });

    test('add and subtract throw on shape mismatch', () {
      final A = Matrix.zeros(2, 2);
      final B = Matrix.zeros(3, 2);

      expect(() => A.add(B), throwsArgumentError);
      expect(() => A.subtract(B), throwsArgumentError);
    });

    test('appendColumn', () {
      final A = Matrix.fromList([
        [1, 2],
        [3, 4],
      ]);

      final B = Matrix.fromList([
        [5],
        [6],
      ]);

      final C = A.appendColumn(B);

      expect(
        C.toList(),
        equals([
          [1, 2, 5],
          [3, 4, 6],
        ]),
      );
    });

    test('appendColumn throws on row mismatch', () {
      final A = Matrix.zeros(2, 2);
      final B = Matrix.zeros(3, 1);

      expect(() => A.appendColumn(B), throwsArgumentError);
    });

    test('diag and diagonal (square)', () {
      final D = Matrix.diag([1, 2, 3]);

      expect(
        D.toList(),
        equals([
          [1, 0, 0],
          [0, 2, 0],
          [0, 0, 3],
        ]),
      );

      expect(D.diagonal(), equals([1, 2, 3]));
    });

    test('diagonal on rectangular matrix', () {
      final A = Matrix.fromList([
        [1, 2, 3],
        [4, 5, 6],
      ]);

      expect(A.diagonal(), equals([1, 5]));
    });

    test('prints matrix with | borders and tab separation', () {
      final A = Matrix.fromList([
        [1.2345, 2.3456],
        [3.4567, 4.5678],
      ]);

      const expected = '| 1.23\t2.35 |\n| 3.46\t4.57 |\n';

      expect(A.toString(), equals(expected));
    });
  });
}
