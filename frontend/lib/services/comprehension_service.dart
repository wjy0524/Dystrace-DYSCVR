// lib/services/comprehension_service.dart

import 'dart:convert';
import 'package:http/http.dart' as http;
import '../model/comprehension.dart';
import '../config/api.dart';

Future<List<ComprehensionItem>> fetchComprehensionMaterial(
    int age,
    String gender,
    String nativeLanguage,
) async {
  final resp = await http.post(
    Uri.parse('$API_BASE_URL/get-comprehension-material'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({
      'age': age,
      'gender': gender,
      'native_language': nativeLanguage,
    }),
  );
  if (resp.statusCode != 200) {
    throw Exception('Failed to load comprehension material');
  }
    // 바디를 UTF8로 디코딩해서 한글 깨짐 방지
  final utf8Body = utf8.decode(resp.bodyBytes);
  final data = jsonDecode(utf8Body) as Map<String, dynamic>;
  return (data['comprehensions'] as List)
      .map((e) => ComprehensionItem.fromJson(e as Map<String, dynamic>))
      .toList();
}