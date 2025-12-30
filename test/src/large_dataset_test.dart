import 'package:multivariate_linear_regression/src/multivariate_linear_regression.dart';
import 'package:test/test.dart';

import '../data/x_train_matrix.dart';
import '../data/x_train_matrix_2.dart';
import '../data/y_train_matrix.dart';
import '../data/y_train_matrix_2.dart';

void main() {
  group('Large Dataset Test', () {
    test('should work with large dataset with intercept', () {
      final mlr = MultivariateLinearRegression(x: xTrainMatrix1, y: yTrainMatrix1);

      expect(
        mlr.predict(xTrainMatrix1[0]).map((e) => e.round()).toList(),
        equals([0]),
      );

      expect(
        mlr.predict(xTrainMatrix1[10]).map((e) => e.round()).toList(),
        equals([434]),
      );

      expect(
        mlr.predict(xTrainMatrix1[20]).map((e) => e.round()).toList(),
        equals([-51]),
      );

      expect(
        mlr.predict(xTrainMatrix1[30]).map((e) => e.round()).toList(),
        equals([525]),
      );

      expect(
        mlr.predict(xTrainMatrix1[40]).map((e) => e.round()).toList(),
        equals([59]),
      );

      expect(
        mlr.predict(xTrainMatrix1[50]).map((e) => e.round()).toList(),
        equals([-36]),
      );
    });

    test('should work with large dataset without intercept', () {
      final testCases = {
        0: 0,
        10: 434,
        20: -51,
        30: 525,
        40: 59,
        50: -36,
        60: 1218,
        164: -71,
      };

      final mlr = MultivariateLinearRegression(x: xTrainMatrix2, y: yTrainMatrix2, intercept: false);

      testCases.forEach((index, expected) {
        final prediction = mlr.predict(xTrainMatrix2[index]).map((e) => e.round()).toList();
        expect(prediction, equals([expected]));
      });
    });
  });
}
