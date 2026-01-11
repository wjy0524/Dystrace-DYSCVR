import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'dyslexia_info_page.dart';
import 'package:google_fonts/google_fonts.dart';
import '../config/api.dart';

class DyslexiaResultPage extends StatefulWidget {
  const DyslexiaResultPage({Key? key}) : super(key: key);

  @override
  State<DyslexiaResultPage> createState() => _DyslexiaResultPageState();
}

class _DyslexiaResultPageState extends State<DyslexiaResultPage> {
  double? probability;
  String riskLevel = "";
  bool isLoading = true;
  String errorMessage = "";

  @override
  void initState() {
    super.initState();
    _calculateAndPredict();
  }

  Future<void> _calculateAndPredict() async {
    try {
      final uid = FirebaseAuth.instance.currentUser!.uid;
      final userDoc = FirebaseFirestore.instance.collection('users').doc(uid);

      // Fetch reading + comprehension results
      final readingSnap = await userDoc
          .collection('reading_results')
          .orderBy('timestamp', descending: true)
          .limit(3)
          .get();
      final compSnap = await userDoc
          .collection('comprehension_results')
          .orderBy('timestamp', descending: true)
          .limit(3)
          .get();

      if (readingSnap.docs.isEmpty || compSnap.docs.isEmpty) {
        setState(() {
          errorMessage = "Not enough data to predict dyslexia risk.";
          isLoading = false;
        });
        return;
      }

      double totalAccuracy = 0, totalWords = 0, totalDuration = 0;
      int totalCorrect = 0, totalQuestions = 0;

      for (var doc in readingSnap.docs) {
        final d = doc.data();
        totalAccuracy += (d['accuracy'] as num).toDouble();
        totalWords += (d['words_read'] as num).toDouble();
        totalDuration += (d['duration_seconds'] as num).toDouble();
      }

      for (var doc in compSnap.docs) {
        final d = doc.data();
        totalCorrect += (d['correct_answers'] as num).toInt();
        totalQuestions += (d['total_questions'] as num).toInt();
      }

      final avgAccuracy = totalAccuracy / readingSnap.docs.length;
      final avgWpm = totalWords / (totalDuration / 60);
      final avgComprehensionRate =
          totalQuestions > 0 ? totalCorrect / totalQuestions : 0;

      final resp = await http.post(
        Uri.parse('$API_BASE_URL/predict_dyslexia'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'wpm': avgWpm,
          'accuracy': avgAccuracy,
          'comprehension_rate': avgComprehensionRate,
        }),
      );

      if (resp.statusCode == 200) {
        final data = jsonDecode(resp.body);
        final prob = (data['probability'] as num).toDouble();

        String level;
        if (prob < 0.33) {
          level = "Low Risk";
        } else if (prob < 0.66) {
          level = "Moderate Risk";
        } else {
          level = "High Risk";
        }

        setState(() {
          probability = prob;
          riskLevel = level;
          isLoading = false;
        });
      } else {
        setState(() {
          errorMessage = "Server error (${resp.statusCode}).";
          isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        errorMessage = "Error: $e";
        isLoading = false;
      });
    }
  }

  IconData _getRiskIcon() {
    switch (riskLevel) {
      case "High Risk":
        return Icons.warning_amber_rounded;
      case "Moderate Risk":
        return Icons.analytics_rounded;
      default:
        return Icons.check_circle_rounded;
    }
  }

  Color _getRiskColor() {
    switch (riskLevel) {
      case "High Risk":
        return Colors.redAccent;
      case "Moderate Risk":
        return Colors.orangeAccent;
      default:
        return Colors.green;
    }
  }

  @override
  Widget build(BuildContext context) {
    const Color bgColor = Color(0xFFF1F8E9);
    const Color mainColor = Color(0xFF66BB6A);

    return Scaffold(
      backgroundColor: bgColor,
      appBar: AppBar(
        centerTitle: true,
        flexibleSpace: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [Color(0xFF81C784), Color(0xFF4CAF50)],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
          ),
        ),
        title: Text(
          "Dyslexia Evaluation Result",
          style: GoogleFonts.montserrat(
            fontSize: 22,
            fontWeight: FontWeight.w700,
            letterSpacing: 1.2,
            color: Colors.white,
          ),
        ),
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24.0),
          child: isLoading
              ? const CircularProgressIndicator()
              : errorMessage.isNotEmpty
                  ? Text(errorMessage,
                      style: const TextStyle(color: Colors.red, fontSize: 16))
                  : AnimatedContainer(
                      duration: const Duration(milliseconds: 600),
                      curve: Curves.easeOut,
                      padding: const EdgeInsets.all(32.0),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(24),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.1),
                            blurRadius: 12,
                            offset: const Offset(0, 6),
                          ),
                        ],
                      ),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(_getRiskIcon(),
                              size: 90, color: _getRiskColor()),
                          const SizedBox(height: 24),
                          const Text(
                            "Predicted Dyslexia Probability",
                            style: TextStyle(
                              fontSize: 22,
                              fontWeight: FontWeight.w600,
                              color: Colors.black87,
                            ),
                          ),
                          const SizedBox(height: 14),
                          Text(
                            "${(probability! * 100).toStringAsFixed(1)}%",
                            style: TextStyle(
                              fontSize: 44,
                              fontWeight: FontWeight.bold,
                              color: _getRiskColor(),
                            ),
                          ),
                          const SizedBox(height: 10),
                          Text(
                            riskLevel,
                            style: TextStyle(
                              fontSize: 24,
                              fontWeight: FontWeight.w700,
                              color: _getRiskColor(),
                            ),
                          ),
                          const SizedBox(height: 40),
                          ElevatedButton.icon(
                            onPressed: () {
                              Navigator.pushReplacement(
                                context,
                                MaterialPageRoute(
                                    builder: (_) => const DyslexiaInfoPage()),
                              );
                            },
                            icon: const Icon(Icons.arrow_forward_ios),
                            label: const Text(
                              "View Dyslexia Resources",
                              style: TextStyle(fontSize: 18),
                            ),
                            style: ElevatedButton.styleFrom(
                              padding: const EdgeInsets.symmetric(
                                  vertical: 16, horizontal: 28),
                              backgroundColor: mainColor,
                              foregroundColor: Colors.white,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                              shadowColor: mainColor.withOpacity(0.4),
                              elevation: 4,
                            ),
                          ),
                        ],
                      ),
                    ),
        ),
      ),
    );
  }
}

