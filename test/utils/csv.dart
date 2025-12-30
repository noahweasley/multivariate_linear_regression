// ignore_for_file: public_member_api_docs

import 'dart:io';

import 'package:csv/csv.dart';

List<List<double>> readCsv(String path) {
  final input = File(path).readAsStringSync();

  final rows = const CsvToListConverter(
    shouldParseNumbers: true,
  ).convert(input);

  return rows.map<List<double>>((row) => row.map<double>((e) => (e as num).toDouble()).toList()).toList();
}
