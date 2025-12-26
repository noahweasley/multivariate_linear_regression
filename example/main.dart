import 'package:multivariate_linear_regression/multivariate_linear_regression.dart';

void main() {
  final x = [
    [0.0, 0.0],
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
  ];

  final y = [
    [0.0, 0.0, 0.0],
    [2.0, 4.0, 3.0],
    [4.0, 6.0, 5.0],
    [6.0, 8.0, 7.0],
  ];

  final mlr = MultivariateLinearRegression(
    x: x,
    y: y,
  );

  // ignore: avoid_print
  print(mlr.predict([3.0, 3.0]));
}
