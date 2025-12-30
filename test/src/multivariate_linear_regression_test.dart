// ignore_for_file: avoid_dynamic_calls

import 'package:multivariate_linear_regression/multivariate_linear_regression.dart';
import 'package:test/test.dart';

import '../data/x02.dart';
import '../data/x42.dart';

void main() {
  group('Multivariate Linear Regression', () {
    test('works with 2 inputs and 3 outputs', () {
      final mlr = MultivariateLinearRegression(
        x: [
          [0, 0],
          [1, 2],
          [2, 3],
          [3, 4],
        ],
        y: [
          [0, 0, 0],
          [2, 4, 3],
          [4, 6, 5],
          [6, 8, 7],
        ],
      );

      final p1 = mlr.predict([2, 3]).map((e) => e.round()).toList();
      final p2 = mlr.predict([4, 4]).map((e) => e.round()).toList();

      expect(p1, [4, 6, 5]);
      expect(p2, [8, 8, 8]);
    });

    test('works with 2 inputs and 3 outputs - intercept is 0', () {
      final mlr = MultivariateLinearRegression(
        x: [
          [0, 0],
          [1, 2],
          [2, 3],
          [3, 4],
        ],
        y: [
          [0, 0, 0],
          [2, 4, 3],
          [4, 6, 5],
          [6, 8, 7],
        ],
      );

      final p1 = mlr.predict([2, 3]).map((e) => e.round()).toList();
      final p2 = mlr.predict([4, 4]).map((e) => e.round()).toList();

      expect(p1, [4, 6, 5]);
      expect(p2, [8, 8, 8]);
    });

    test('works with 2 inputs and 3 outputs - intercept is not 0', () {
      final mlr = MultivariateLinearRegression(
        x: [
          [0, 0],
          [1, 2],
          [2, 3],
          [3, 4],
        ],
        y: [
          [-1, 2, 10],
          [1, 6, 13],
          [3, 8, 15],
          [5, 10, 17],
        ],
      );

      final p1 = mlr.predict([2, 3]).map((e) => e.round()).toList();
      final p2 = mlr.predict([4, 4]).map((e) => e.round()).toList();

      expect(p1, [3, 8, 15]);
      expect(p2, [7, 10, 18]);
    });

    test('works with 2 inputs and 1 output (x02)', () {
      final x02X = x02Data['x']!;
      final x02Y = x02Data['y']!;

      final mlr = MultivariateLinearRegression(x: x02X, y: x02Y, intercept: false);
      final prediction = mlr.predictBatch(x02X);

      expect(prediction[0][0], closeTo(38.05, 0.01));
    });

    test('works with 2 inputs and 1 output (x42)', () {
      final x42X = x42Data['x']!;
      final x42Y = x42Data['y']!;

      final expectedWeights = [
        83.125,
        2.625,
        3.125,
        3.75,
        -2.0,
        -4.375,
        0.0,
        1.5,
        -0.25,
      ];

      final mlr = MultivariateLinearRegression(x: x42X, y: x42Y, intercept: false);
      final weights = mlr.weights.map((row) => row.toList()).toList();

      for (var i = 0; i < weights.length; i++) {
        expect(weights[i][0], closeTo(expectedWeights[i], 1e-6));
      }
    });

    test('toJson and load', () {
      final mlr = MultivariateLinearRegression(
        x: [
          [0, 0],
          [1, 2],
          [2, 3],
          [3, 4],
        ],
        y: [
          [0, 0, 0],
          [2, 4, 3],
          [4, 6, 5],
          [6, 8, 7],
        ],
      );

      // final json = mlr.toJson();
      final loaded = MultivariateLinearRegression.load(mlr);

      final p = loaded.predict([2, 3]).map((e) => e.round()).toList();
      expect(p, [4, 6, 5]);
    });

    test('data mining test 1-1', () {
      final X = [
        [4.47],
        [208.3],
        [3400.0],
      ];

      final Y = [
        [0.51],
        [105.66],
        [1800.0],
      ];

      final mlr = MultivariateLinearRegression(x: X, y: Y);
      final weights = mlr.weights.map((row) => row.toList()).toList();

      expect(weights[0][0], closeTo(0.53, 0.01));
      expect(weights[1][0], closeTo(-3.29, 0.01));
    });

    test('data mining test 1-2', () {
      final X = <List<double>>[
        [4.47, 1],
        [208.3, 1],
        [3400.0, 1],
      ];

      final Y = <List<double>>[
        [0.51],
        [105.66],
        [1800.0],
      ];

      final mlr = MultivariateLinearRegression(x: X, y: Y, intercept: false);
      final weights = mlr.weights.map((row) => row.toList()).toList();

      expect(weights[0][0], closeTo(0.53, 0.01));
      expect(weights[1][0], closeTo(-3.29, 0.01));
    });

    test('data mining test 2', () {
      final X = <List<double>>[
        [1, 1, 1],
        [2, 1, 1],
        [3, 1, 1],
      ];

      final Y = <List<double>>[
        [2, 3],
        [4, 6],
        [6, 9],
      ];

      final mlr = MultivariateLinearRegression(x: X, y: Y);
      final weights = mlr.weights.map((row) => row.toList()).toList();

      expect(weights[0][0], closeTo(2, 0.01));
      expect(weights[0][1], closeTo(3, 0.01));
      expect(weights[1][0], closeTo(0, 0.01));
      expect(weights[1][1], closeTo(0, 0.01));
      expect(weights[2][0], closeTo(0, 0.01));
      expect(weights[2][1], closeTo(0, 0.01));
    });

    test('data mining statistics test', () {
      final X = <List<double>>[
        [3, 1],
        [4, 2],
        [10, 3],
        [6, 4],
        [7, 5],
      ];

      final Y = <List<double>>[
        [19],
        [28],
        [37],
        [46],
        [40],
      ];

      final mlr = MultivariateLinearRegression(x: X, y: Y);
      final json = mlr.toJson();
      final summary = json['summary'];

      expect(summary!['regressionStatistics']['standardError'], closeTo(6.27, 0.1));

      final vars = summary['variables'] as List;
      expect(vars[0]['coefficients'][0], closeTo(0.75, 0.01));
      expect(vars[0]['standardError'], closeTo(1.4, 0.01));
      expect(vars[0]['tStat'], closeTo(0.53, 0.01));

      expect(vars[1]['coefficients'][0], closeTo(5.25, 0.01));
      expect(vars[1]['standardError'], closeTo(2.43, 0.01));
      expect(vars[1]['tStat'], closeTo(2.16, 0.01));
      expect(vars[2]['label'], 'Intercept');
      expect(vars[2]['coefficients'][0], closeTo(13.75, 0.01));
      expect(vars[2]['standardError'], closeTo(7.81, 0.01));
      expect(vars[2]['tStat'], closeTo(1.76, 0.01));
    });

    test('statistics can be disabled', () {
      final X = <List<double>>[
        [3, 1],
        [4, 2],
        [10, 3],
        [6, 4],
        [7, 5],
      ];

      final Y = <List<double>>[
        [19],
        [28],
        [37],
        [46],
        [40],
      ];

      final mlr = MultivariateLinearRegression(x: X, y: Y, statistics: false);
      final json = mlr.toJson();

      expect(json['summary'], isNull);
    });

    test('throws on wrong data type', () {
      final X = <List<double>>[
        [3, 1],
        [4, 2],
        [10, 3],
        [6, 4],
        [7, 5],
      ];

      final Y = <List<double>>[
        [19],
        [28],
        [37],
        [46],
        [40],
      ];

      final mlr = MultivariateLinearRegression(x: X, y: Y);

      expect(() => mlr.predict([3]), throwsArgumentError);
    });
  });
}
