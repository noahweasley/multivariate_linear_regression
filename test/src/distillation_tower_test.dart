import 'package:multivariate_linear_regression/multivariate_linear_regression.dart';
import 'package:test/test.dart';

import '../utils/csv.dart';

void main() {
  group('Distillation Tower Test', () {
    late List<List<double>> x;
    late List<List<double>> y;

    setUp(() {
      x = readCsv('test/data/x_distillation.csv');
      y = readCsv('test/data/y_distillation.csv');
    });

    test('should work with large dataset with intercept', () {
      final mlr = MultivariateLinearRegression(x: x, y: y);

      expect(
        mlr.predict(x[0]).map((e) => e.round()).toList(),
        equals([33]),
      );

      expect(
        mlr.predict(x[10]).map((e) => e.round()).toList(),
        equals([35]),
      );

      expect(
        mlr.predict(x[20]).map((e) => e.round()).toList(),
        equals([41]),
      );
      expect(
        mlr.predict(x[30]).map((e) => e.round()).toList(),
        equals([33]),
      );

      expect(
        mlr.predict(x[40]).map((e) => e.round()).toList(),
        equals([46]),
      );

      expect(
        mlr.predict(x[50]).map((e) => e.round()).toList(),
        equals([33]),
      );
      expect(
        mlr.predict(x[60]).map((e) => e.round()).toList(),
        equals([35]),
      );

      expect(
        mlr.predict(x[250]).map((e) => e.round()).toList(),
        equals([36]),
      );
    });

    test('should work with large dataset without intercept', () {
      final mlr = MultivariateLinearRegression(x: x, y: y, intercept: false);

      expect(
        mlr.predict(x[0]).map((e) => e.round()).toList(),
        equals([33]),
      );

      expect(
        mlr.predict(x[10]).map((e) => e.round()).toList(),
        equals([34]),
      );

      expect(
        mlr.predict(x[20]).map((e) => e.round()).toList(),
        equals([41]),
      );

      expect(
        mlr.predict(x[30]).map((e) => e.round()).toList(),
        equals([33]),
      );
      expect(
        mlr.predict(x[40]).map((e) => e.round()).toList(),
        equals([45]),
      );

      expect(
        mlr.predict(x[50]).map((e) => e.round()).toList(),
        equals([33]),
      );

      expect(
        mlr.predict(x[60]).map((e) => e.round()).toList(),
        equals([35]),
      );

      expect(
        mlr.predict(x[250]).map((e) => e.round()).toList(),
        equals([36]),
      );
    });
  });
}
