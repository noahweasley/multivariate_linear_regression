/// A lightweight 2D matrix implementation designed specifically for
/// multivariate linear regression and SVD-based pseudoinverse.
///
/// This class intentionally implements only the subset of linear algebra
/// operations required
///
/// Design goals:
/// - Supports rectangular (m × n) matrices
/// - No assumption of squareness
/// - JS `ml-matrix` parity
class Matrix {
  /// Number of rows (m)
  final int rows;

  /// Number of columns (n)
  final int cols;

  /// Internal row-major storage
  ///
  /// `_d[i][j]` represents the element at row `i`, column `j`.
  final List<List<double>> _d;

  /// Internal constructor.
  ///
  /// Assumes the data is already validated and rectangular.
  Matrix._(this._d)
      : rows = _d.length,
        cols = _d.isEmpty ? 0 : _d[0].length;

  /// Creates a matrix from a 2D list.
  ///
  /// Throws an [ArgumentError] if the input is jagged (rows with different lengths).
  ///
  /// Example:
  /// ```dart
  /// final A = Matrix.fromList([
  ///   [1, 2],
  ///   [3, 4],
  /// ]);
  /// ```
  factory Matrix.fromList(List<List<double>> data) {
    if (data.isEmpty) return Matrix._([]);

    final c = data[0].length;

    for (final row in data) {
      if (row.length != c) {
        throw ArgumentError('Jagged matrix is not allowed');
      }
    }

    // Defensive copy
    return Matrix._(List.generate(data.length, (i) => List.of(data[i])));
  }

  /// Creates an `r × c` matrix filled with zeros.
  ///
  /// Commonly used to:
  /// - initialize result matrices
  /// - build identity or diagonal matrices
  factory Matrix.zeros(int r, int c) {
    return Matrix._(
      List.generate(r, (_) => List.filled(c, 0.0)),
    );
  }

  /// Creates an `n × n` identity matrix.
  ///
  /// Identity matrix `I` satisfies:
  /// ```
  /// A × I = I × A = A
  /// ```
  factory Matrix.identity(int n) {
    final m = Matrix.zeros(n, n);
    for (var i = 0; i < n; i++) {
      m._d[i][i] = 1.0;
    }
    return m;
  }

  /// Returns the element at row `r`, column `c`.
  double get(int r, int c) => _d[r][c];

  /// Sets the element at row `r`, column `c`.
  ///
  /// This is intentionally explicit to avoid accidental mutation.
  void set(int r, int c, double value) {
    _d[r][c] = value;
  }

  /// Returns a deep copy of the matrix data.
  List<List<double>> toList() {
    return List.generate(rows, (i) => List.of(_d[i]));
  }

  /// Returns the transpose of this matrix.
  ///
  /// If this matrix is `m × n`, the result is `n × m`.
  ///
  /// Transpose is essential for:
  /// - Normal equations (`XᵀX`)
  /// - SVD (`Uᵀ`, `Vᵀ`)
  /// - Pseudoinverse (`V Σ⁺ Uᵀ`)
  Matrix transpose() {
    final t = Matrix.zeros(cols, rows);

    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < cols; j++) {
        t._d[j][i] = _d[i][j];
      }
    }

    return t;
  }

  /// Matrix multiplication.
  ///
  /// Multiplies this matrix (`m × k`) with matrix `B` (`k × n`)
  /// producing a matrix of shape (`m × n`).
  Matrix multiply(Matrix B) {
    if (cols != B.rows) {
      throw ArgumentError(
        'Shape mismatch: $rows x $cols cannot multiply ${B.rows} x ${B.cols}',
      );
    }

    final C = Matrix.zeros(rows, B.cols);

    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < B.cols; j++) {
        double sum = 0.0;

        for (var k = 0; k < cols; k++) {
          sum += _d[i][k] * B._d[k][j];
        }

        C._d[i][j] = sum;
      }
    }

    return C;
  }

  /// Matrix subtraction.
  Matrix subtract(Matrix B) {
    if (rows != B.rows || cols != B.cols) {
      throw ArgumentError('Shape mismatch for subtraction');
    }

    final C = Matrix.zeros(rows, cols);

    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < cols; j++) {
        C._d[i][j] = _d[i][j] - B._d[i][j];
      }
    }

    return C;
  }

  /// Concatenates another matrix to the right (column-wise).
  Matrix appendColumn(Matrix B) {
    if (rows != B.rows) {
      throw ArgumentError('Row count mismatch in column concatenation');
    }

    final C = Matrix.zeros(rows, cols + B.cols);

    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < cols; j++) {
        C._d[i][j] = _d[i][j];
      }

      for (var j = 0; j < B.cols; j++) {
        C._d[i][cols + j] = B._d[i][j];
      }
    }

    return C;
  }

  /// Creates a diagonal matrix from a list of values.
  static Matrix diag(List<double> values) {
    final D = Matrix.zeros(values.length, values.length);

    for (var i = 0; i < values.length; i++) {
      D._d[i][i] = values[i];
    }

    return D;
  }

  /// Extracts the main diagonal as a list.
  List<double> diagonal() {
    final n = rows < cols ? rows : cols;

    return List.generate(n, (i) => _d[i][i]);
  }

  /// Clones the current Matrix
  Matrix clone() {
    return Matrix.fromList(this.toList());
  }

  /// Multiplies every element of the matrix by a scalar.
  Matrix scale(double scalar) {
    final C = Matrix.zeros(rows, cols);

    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < cols; j++) {
        C._d[i][j] = _d[i][j] * scalar;
      }
    }

    return C;
  }

  /// Adds another matrix element-wise.
  ///
  /// Example:
  /// ```
  /// C = A.add(B)
  /// ```
  Matrix add(Matrix B) {
    if (rows != B.rows || cols != B.cols) {
      throw ArgumentError('Shape mismatch for addition');
    }

    final C = Matrix.zeros(rows, cols);

    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < cols; j++) {
        C._d[i][j] = _d[i][j] + B._d[i][j];
      }
    }

    return C;
  }

  /// Negates the matrix (equivalent to multiplying by -1).
  ///
  /// Example:
  /// ```
  /// C = A.neg()
  /// ```
  Matrix neg() {
    return this.scale(-1.0);
  }
}
