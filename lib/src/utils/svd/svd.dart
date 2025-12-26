// ignore_for_file: public_member_api_docs
import 'dart:math';

import 'package:multivariate_linear_regression/src/utils/constants.dart';
import 'package:multivariate_linear_regression/src/utils/svd/matrix.dart';
import 'package:multivariate_linear_regression/src/utils/svd/svd_result.dart';
import 'package:multivariate_linear_regression/src/utils/svd/utils.dart';

class GolubReinschSVD {
  final SVDResult results;

  /// private constructor use `GolubReinschSVD.decompose()`
  GolubReinschSVD._({required this.results});

  factory GolubReinschSVD.decompose(Matrix A) {
    final eps = epsilon;
    final m = A.rows;
    final n = A.cols;

    if (m == 0 || n == 0) {
      throw ArgumentError('Matrix must be non-empty');
    }

    final a = A.clone();
    final nu = min(m, n);
    final ni = min(m + 1, n);

    final s = List<double>.filled(ni, 0.0);
    final e = List<double>.filled(n, 0.0);
    final work = List<double>.filled(m, 0.0);

    final U = Matrix.zeros(m, nu);
    final V = Matrix.zeros(n, n);

    final nct = min(m - 1, n);
    final nrt = max(0, min(n - 2, m));
    final mrc = max(nct, nrt);

    for (var k = 0; k < mrc; k++) {
      if (k < nct) {
        s[k] = 0.0;

        for (var i = k; i < m; i++) {
          s[k] = hypotenuse(s[k], a.get(i, k));
        }

        if (s[k] != 0.0) {
          if (a.get(k, k) < 0) s[k] = -s[k];

          for (var i = k; i < m; i++) {
            a.set(i, k, a.get(i, k) / s[k]);
          }

          a.set(k, k, a.get(k, k) + 1);
        }

        s[k] = -s[k];
      }

      for (var j = k + 1; j < n; j++) {
        if (k < nct && s[k] != 0) {
          var t = 0.0;

          for (var i = k; i < m; i++) {
            t += a.get(i, k) * a.get(i, j);
          }

          t = -t / a.get(k, k);

          for (var i = k; i < m; i++) {
            a.set(i, j, a.get(i, j) + t * a.get(i, k));
          }
        }

        e[j] = a.get(k, j);
      }

      if (k < nct) {
        for (var i = k; i < m; i++) {
          U.set(i, k, a.get(i, k));
        }
      }

      if (k < nrt) {
        e[k] = 0.0;

        for (var i = k + 1; i < n; i++) {
          e[k] = hypotenuse(e[k], e[i]);
        }

        if (e[k] != 0.0) {
          // 0 - e[k] is still the same as -e[k] :)
          if (e[k + 1] < 0) e[k] = 0 - e[k];

          for (var i = k + 1; i < n; i++) {
            e[i] /= e[k];
          }

          e[k + 1] += 1;
        }

        e[k] = -e[k];

        if (k + 1 < m && e[k] != 0) {
          for (var i = k + 1; i < m; i++) {
            work[i] = 0.0;
          }

          for (var i = k + 1; i < m; i++) {
            for (var j = k + 1; j < n; j++) {
              work[i] += e[j] * a.get(i, j);
            }
          }

          for (var j = k + 1; j < n; j++) {
            final t = -e[j] / e[k + 1];

            for (var i = k + 1; i < m; i++) {
              a.set(i, j, a.get(i, j) + t * work[i]);
            }
          }
        }

        for (var i = k + 1; i < n; i++) {
          V.set(i, k, e[i]);
        }
      }
    }

    var p = min(n, m + 1);

    if (nct < n) s[nct] = a.get(nct, nct);
    if (m < p) s[p - 1] = 0.0;
    if (nrt + 1 < p) e[nrt] = a.get(nrt, p - 1);

    e[p - 1] = 0.0;

    for (var j = nct; j < nu; j++) {
      for (var i = 0; i < m; i++) {
        U.set(i, j, 0.0);
      }

      U.set(j, j, 1.0);
    }

    for (var k = nct - 1; k >= 0; k--) {
      if (s[k] != 0.0) {
        for (var j = k + 1; j < nu; j++) {
          var t = 0.0;

          for (var i = k; i < m; i++) {
            t += U.get(i, k) * U.get(i, j);
          }

          t = -t / U.get(k, k);

          for (var i = k; i < m; i++) {
            U.set(i, j, U.get(i, j) + t * U.get(i, k));
          }
        }

        for (var i = k; i < m; i++) {
          U.set(i, k, -U.get(i, k));
        }

        U.set(k, k, 1.0 + U.get(k, k));

        for (var i = 0; i < k - 1; i++) {
          U.set(i, k, 0.0);
        }
      } else {
        for (var i = 0; i < m; i++) {
          U.set(i, k, 0.0);
        }

        U.set(k, k, 1.0);
      }
    }

    for (var k = n - 1; k >= 0; k--) {
      if (k < nrt && e[k] != 0) {
        for (var j = k + 1; j < n; j++) {
          var t = 0.0;

          for (var i = k + 1; i < n; i++) {
            t += V.get(i, k) * V.get(i, j);
          }

          t = -t / V.get(k + 1, k);

          for (var i = k + 1; i < n; i++) {
            V.set(i, j, V.get(i, j) + t * V.get(i, k));
          }
        }
      }

      for (var i = 0; i < n; i++) {
        V.set(i, k, 0);
      }

      V.set(k, k, 1);
    }

    final pp = p - 1;
    var iter = 0;

    while (p > 0) {
      int k, kase;

      for (k = p - 2; k >= -1; k--) {
        if (k == -1) {
          break;
        }

        final alpha = double.minPositive + eps * (s[k] + s[k + 1].abs()).abs();

        if (e[k].abs() <= alpha || e[k].isNaN) {
          e[k] = 0.0;
          break;
        }
      }

      if (k == p - 2) {
        kase = 4;
      } else {
        int ks;

        for (ks = p - 1; ks >= k; ks--) {
          if (ks == k) {
            break;
          }

          final t = (ks != p ? e[ks].abs() : 0.0) + (ks != k + 1 ? e[ks - 1].abs() : 0.0);

          if (s[ks].abs() <= eps * t) {
            s[ks] = 0.0;
            break;
          }
        }

        if (ks == k) {
          kase = 3;
        } else if (ks == p - 1) {
          kase = 1;
        } else {
          kase = 2;
          k = ks;
        }
      }

      k++;

      switch (kase) {
        case 1:
          {
            var f = e[p - 2];
            e[p - 2] = 0.0;

            for (var j = p - 2; j >= k; j--) {
              var t = hypotenuse(s[j], f);
              final cs = s[j] / t;
              final sn = f / t;

              s[j] = t;

              if (j != k) {
                f = -sn * e[j - 1];
                e[j - 1] = cs * e[j - 1];
              }

              for (var i = 0; i < n; i++) {
                t = cs * V.get(i, j) + sn * V.get(i, p - 1);
                V.set(i, p - 1, -sn * V.get(i, j) + cs * V.get(i, p - 1));
                V.set(i, j, t);
              }
            }
            break;
          }

        case 2:
          {
            var f = e[k - 1];
            e[k - 1] = 0.0;

            for (var j = k; j < p; j++) {
              var t = hypotenuse(s[j], f);
              final cs = s[j] / t;
              final sn = f / t;

              s[j] = t;
              f = -sn * e[j];
              e[j] = cs * e[j];

              for (var i = 0; i < m; i++) {
                t = cs * U.get(i, j) + sn * U.get(i, k - 1);
                U.set(i, k - 1, -sn * U.get(i, j) + cs * U.get(i, k - 1));
                U.set(i, j, t);
              }
            }
            break;
          }

        case 3:
          {
            final stepOne = max(s[p - 1].abs(), s[p - 2].abs());
            final stepTwo = e[p - 2].abs();
            final stepThree = max(stepOne, stepTwo);
            final stepFour = max(s[k].abs(), e[k].abs());

            final scale = max(stepThree, stepFour);

            final sp = s[p - 1] / scale;
            final spm1 = s[p - 2] / scale;
            final epm1 = e[p - 2] / scale;
            final sk = s[k] / scale;
            final ek = e[k] / scale;

            final b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
            final c = sp * epm1 * (sp * epm1);

            var shift = 0.0;

            if (b != 0.0 || c != 0.0) {
              final bbc = b * b + c;

              shift = b < 0 ? (0 - sqrt(bbc)) : sqrt(bbc);
              shift = c / (b + shift);
            }

            var f = (sk + sp) * (sk - sp) + shift;
            var g = sk * ek;

            for (var j = k; j < p - 1; j++) {
              var t = hypotenuse(f, g);
              if (t == 0.0) t = double.minPositive;

              var cs = f / t;
              var sn = g / t;
              if (j != k) e[j - 1] = t;

              f = cs * s[j] + sn * e[j];
              e[j] = cs * e[j] - sn * s[j];
              g = sn * s[j + 1];
              s[j + 1] = cs * s[j + 1];

              for (var i = 0; i < n; i++) {
                t = cs * V.get(i, j) + sn * V.get(i, j + 1);
                V.set(i, j + 1, -sn * V.get(i, j) + cs * V.get(i, j + 1));
                V.set(i, j, t);
              }

              t = hypotenuse(f, g);

              if (t == 0.0) t = double.minPositive;
              cs = f / t;
              sn = g / t;

              s[j] = t;
              f = cs * e[j] + sn * s[j + 1];
              s[j + 1] = -sn * e[j] + cs * s[j + 1];
              g = sn * e[j + 1];
              e[j + 1] = cs * e[j + 1];

              if (j < m - 1) {
                for (var i = 0; i < m; i++) {
                  t = cs * U.get(i, j) + sn * U.get(i, j + 1);
                  U.set(i, j + 1, -sn * U.get(i, j) + cs * U.get(i, j + 1));
                  U.set(i, j, t);
                }
              }
            }

            e[p - 2] = f;
            iter = iter + 1;
            break;
          }

        case 4:
          {
            if (s[k] <= 0.0) {
              s[k] = s[k] < 0.0 ? -s[k] : 0.0;

              for (var i = 0; i <= pp; i++) {
                V.set(i, k, -V.get(i, k));
              }
            }

            while (k < pp) {
              if (s[k] >= s[k + 1]) break;

              var t = s[k];
              s[k] = s[k + 1];
              s[k + 1] = t;

              if (k < n - 1) {
                for (int i = 0; i < n; i++) {
                  t = V.get(i, k + 1);
                  V.set(i, k + 1, V.get(i, k));
                  V.set(i, k, t);
                }
              }

              if (k < m - 1) {
                for (var i = 0; i < m; i++) {
                  t = U.get(i, k + 1);
                  U.set(i, k + 1, U.get(i, k));
                  U.set(i, k, t);
                }
              }
              k++;
            }

            iter = 0;
            p--;
            break;
          }
      }
    }

    final results = SVDResult(
      rows: m,
      cols: n,
      leftSingleVectors: U,
      rightSingularVectors: V,
      singularValues: s,
    );

    return GolubReinschSVD._(results: results);
  }
}
