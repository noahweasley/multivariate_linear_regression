// ignore_for_file: public_member_api_docs
import 'dart:math';

double hypotenuse(double a, double b) {
  if (a.abs() > b.abs()) {
    final r = b / a;

    return a.abs() * sqrt(1 + r * r);
  }

  if (b != 0.0) {
    final r = a / b;

    return b.abs() * sqrt(1 + r * r);
  }

  return 0.0;
}
