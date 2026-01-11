// English UI version of ReadingSpeedPage

import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:js/js_util.dart' as js_util;
import 'js_interop.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'history_page.dart';
import 'package:flutter_widget_from_html_core/flutter_widget_from_html_core.dart';
import 'comprehension_page.dart';
import '../config/api.dart';


class WordMetrics {
  final int fixationCount;
  final double avgFixationDuration;
  final int regressionCount;

  WordMetrics({
    required this.fixationCount,
    required this.avgFixationDuration,
    required this.regressionCount,
  });

  static final zero = WordMetrics(
    fixationCount: 0,
    avgFixationDuration: 0.0,
    regressionCount: 0,
  );
}

// Compute word-based gaze metrics
WordMetrics computeWordBasedMetrics(
  List<Map<String, dynamic>> eyeData,
  List<Rect> wordRects,
) {
  if (eyeData.isEmpty || wordRects.isEmpty) return WordMetrics.zero;

  int fixationCount = 0;
  int regressionCount = 0;
  double totalFixDur = 0;

  int? prevBox;
  int prevTime = eyeData.first['t'] as int;

  for (final sample in eyeData) {
    final x = sample['x'] as double;
    final y = sample['y'] as double;
    final t = sample['t'] as int;

    final idx = wordRects.indexWhere((r) => r.contains(Offset(x, y)));

    if (idx == prevBox) {
      totalFixDur += (t - prevTime);
    } else {
      if (prevBox != null) fixationCount++;
      if (prevBox != null && idx < prevBox) regressionCount++;
      prevBox = idx >= 0 ? idx : null;
      prevTime = t;
    }
  }

  final avgFixDur = fixationCount > 0 ? totalFixDur / fixationCount : 0.0;
  return WordMetrics(
    fixationCount: fixationCount,
    avgFixationDuration: avgFixDur,
    regressionCount: regressionCount,
  );
}

class ReadingSpeedPage extends StatefulWidget {
  final String participantName;
  final int participantAge;
  final String participantGender;

  const ReadingSpeedPage({
    Key? key,
    required this.participantName,
    required this.participantAge,
    required this.participantGender,
  }) : super(key: key);

  @override
  State<ReadingSpeedPage> createState() => _ReadingSpeedPageState();
}

class _ReadingSpeedPageState extends State<ReadingSpeedPage> {
  bool _isRecording = false;
  bool _isUploading = false;
  int _elapsedSeconds = 0;
  Stopwatch _stopwatch = Stopwatch();

  List<String> _passages = [];
  int _currentIndex = 0;
  bool _hasRecorded = false;

  @override
  void initState() {
    _removeAllGazeIndicators();
    resetWebGazer();
    super.initState();
    _fetchPassages();
  }

  String _buildHtmlFromText(String text) {
    final tokens = text.split(RegExp(r'\s+'));
    return tokens
        .map((t) => '<span class="word">${HtmlEscape().convert(t)}</span>')
        .join(' ');
  }

  Future<void> _fetchPassages() async {
    try {
      final resp = await http.post(
        Uri.parse('$API_BASE_URL/get-passages'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'age': widget.participantAge,
          'gender': widget.participantGender,
          'native_language': 'english',
        }),
      );
      if (resp.statusCode == 200) {
        final utf8Body = utf8.decode(resp.bodyBytes);
        final data = json.decode(utf8Body) as Map<String, dynamic>;
        final raw = data['passages'];
        setState(() {
          _passages = (raw as List)
            .map((e) {
              if (e is Map && e.containsKey('title') && e.containsKey('content')) {
              return e['content'] as String;
            } else if (e is Map && e.containsKey('text')) {
              return e['text'] as String;
            } else {
              return e.toString();
            }
          })
          .toList();
          _currentIndex = 0;
          _hasRecorded = false;
        });
      } else {
        _showErrorDialog('Failed to load passages: ${resp.statusCode}');
      }
    } catch (e) {
      _showErrorDialog('Error loading passages: $e');
    }
  }

  Future<void> _startRecording() async {
    try {
      resetWebGazer();
      js_util.callMethod(js_util.globalThis, 'collectWordBoxes', []);
      await startEyeTracking();
      await _showCalibrationDialogOverTextBox();
      await startMicRecording();

      _stopwatch.reset();
      _stopwatch.start();
      setState(() {
        _isRecording = true;
        _elapsedSeconds = 0;
      });

      await Future.doWhile(() async {
        await Future.delayed(const Duration(seconds: 1));
        if (!_stopwatch.isRunning) return false;
        setState(() => _elapsedSeconds++);
        return true;
      });
    } catch (e) {
      _showErrorDialog('Failed to start recording: $e');
    }
  }

  void _removeAllGazeIndicators() {
    js_util.callMethod(js_util.globalThis, 'eval', ['''
      document.querySelectorAll('#gazeIndicator').forEach(el => el.remove());
      window._gazeIndicator = null;
    ''']);
  }

  Future<void> _stopRecording() async {
    _stopwatch.stop();
    setState(() {
      _isRecording = false;
      _isUploading = true;
    });
    try {
      final bytes = await stopMicRecording();
      final eyeData = await stopEyeTracking();

      final wordRects = await getWordBoxes();
      final wordMetrics = computeWordBasedMetrics(eyeData, wordRects);

      await _sendToBackend(
        bytes,
        eyeData,
        wordMetrics: wordMetrics,
      );
      setState(() => _hasRecorded = true);
    } catch (e) {
      _showErrorDialog('Failed to stop recording: $e');
    } finally {
      setState(() => _isUploading = false);
    }
  }

  Future<void> _sendToBackend(
    Uint8List audioBytes,
    List<dynamic> eyeData, {
    required WordMetrics wordMetrics,
  }) async {
    final uri = Uri.parse('$API_BASE_URL/reading_test');
    final request = http.MultipartRequest('POST', uri)
      ..fields['expected'] = _passages[_currentIndex]
      ..fields['eye_data'] = jsonEncode(eyeData)
      ..fields['fixation_count'] = wordMetrics.fixationCount.toString()
      ..fields['avg_fixation_dur_ms'] =
          wordMetrics.avgFixationDuration.toStringAsFixed(0)
      ..fields['regression_count'] = wordMetrics.regressionCount.toString()
      ..files.add(http.MultipartFile.fromBytes(
        'audio',
        audioBytes,
        filename: 'recording.webm',
        contentType: MediaType('audio', 'webm'),
      ));

    final response = await request.send();
    final body = await response.stream.bytesToString();
    if (response.statusCode == 200) {
      final result = jsonDecode(body) as Map<String, dynamic>;
      final accuracy = (result['accuracy'] as num).toDouble();
      final wordsRead = (result['words_read'] as num).toInt();
      final durationSeconds = (result['duration_seconds'] as num).toInt();

      final uid = FirebaseAuth.instance.currentUser!.uid;
      await FirebaseFirestore.instance
          .collection('users')
          .doc(uid)
          .collection('reading_results')
          .add({
        'accuracy': accuracy,
        'words_read': wordsRead,
        'duration_seconds': durationSeconds,
        'timestamp': FieldValue.serverTimestamp(),
      });

      _showEyeResultDialog();
    } else {
      _showErrorDialog('Upload failed: ${response.statusCode}');
    }
  }

  void _showEyeResultDialog() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => AlertDialog(
        content: const Padding(
          padding: EdgeInsets.symmetric(vertical: 24, horizontal: 16),
          child: Text(
            'Great job!\n\nPlease tap the button below to continue.',
            textAlign: TextAlign.center,
          ),
        ),
        actionsPadding:
            const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
        actions: [
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: _goToNextRound,
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF9575CD),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(24),
                ),
                elevation: 0,
              ),
              child: Text(
                _currentIndex < _passages.length - 1
                    ? 'Read Next Passage'
                    : 'Proceed to Comprehension Test',
                style: const TextStyle(fontSize: 16),
              ),
            ),
          ),
        ],
      ),
    );
  }

  void _goToNextRound() {
    _removeAllGazeIndicators();
    resetWebGazer();

    Navigator.pop(context);
    if (_currentIndex < _passages.length - 1) {
      setState(() {
        _currentIndex++;
        _elapsedSeconds = 0;
        _hasRecorded = false;
      });
    } else {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => ComprehensionPage(
            age: widget.participantAge,
            gender: widget.participantGender,
            nativeLanguage: 'english',
          ),
        ),
      );
    }
  }

  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Error'),
        content: Text(message),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('OK'))
        ],
      ),
    );
  }

  final GlobalKey _textBoxKey = GlobalKey();

  Future<void> _showCalibrationDialogOverTextBox() async {
    final renderBox = _textBoxKey.currentContext!.findRenderObject() as RenderBox;
    final rect = renderBox.localToGlobal(Offset.zero) & renderBox.size;

    const int gridSize = 3;
    final points = [
      for (int r = 0; r < gridSize; r++)
        for (int c = 0; c < gridSize; c++)
          Offset(
            rect.left + rect.width * c / (gridSize - 1),
            rect.top + rect.height * r / (gridSize - 1),
          ),
    ];

    int idx = 0;
    await showDialog(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setSt) {
          return Stack(
            children: [
              GestureDetector(onTap: () {}, child: Container(color: Colors.black45)),
              Positioned(
                left: points[idx].dx - 24,
                top: points[idx].dy - 24,
                child: GestureDetector(
                  onTap: () {
                    if (idx < points.length - 1) {
                      setSt(() => idx++);
                    } else {
                      Navigator.of(ctx).pop();
                    }
                  },
                  child: Container(
                    width: 48,
                    height: 48,
                    decoration: BoxDecoration(
                      color: Colors.blueAccent.withOpacity(0.9),
                      shape: BoxShape.circle,
                    ),
                    alignment: Alignment.center,
                    child: const Text('●',
                        style: TextStyle(color: Colors.white, fontSize: 24)),
                  ),
                ),
              ),
              Positioned(
                left: rect.left,
                top: rect.bottom + 8,
                width: rect.width,
                child: Text(
                  'Look at and click the dot (${idx + 1}/${points.length})',
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.white, fontSize: 16),
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  void _showHelpDialog() {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Help'),
        content: const Text(
          "1. Calibration\n"
          "Tap each white-blue circle in order to calibrate your gaze.\n"
          "When the red dot aligns with the white dot, tap it to record an accurate gaze position.\n\n"
          "2. Reading Test\n"
          "The reading test consists of 3 passages.\n"
          "Press 'Start' to begin reading aloud. Press 'Stop' to end.\n\n"
          "3. Comprehension Test\n"
          "After reading all passages, you will take a simple multiple-choice comprehension test.\n"
          "Each passage includes two comprehension questions.\n\n"
          "4. View Results\n"
          "Once all steps are completed, you can view your past results as charts and tables in the 'History' page.",
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_passages.isEmpty) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    final currentText = _passages[_currentIndex];

    return Scaffold(
      appBar: AppBar(
        title: Text('Passage ${_currentIndex + 1}/${_passages.length}'),
        backgroundColor: const Color(0xFF81C784),
        actions: [
          IconButton(
            icon: const Icon(Icons.help_outline),
            tooltip: 'Help',
            onPressed: _showHelpDialog,
          ),
          IconButton(
            icon: const Icon(Icons.history),
            tooltip: 'View History',
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => const HistoryPage()),
              );
            },
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Expanded(
              child: Container(
                key: _textBoxKey,
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.white,
                  border: Border.all(color: Colors.grey.shade300),
                  borderRadius: BorderRadius.circular(16),
                  boxShadow: const [
                    BoxShadow(
                      color: Colors.black12,
                      blurRadius: 10,
                      offset: Offset(0, 5),
                    ),
                  ],
                ),
                child: SingleChildScrollView(
                  child: HtmlWidget(
                    _buildHtmlFromText(currentText),
                    customStylesBuilder: (element) {
                      if (element.classes.contains('word')) {
                        return {'display': 'inline-block', 'padding': '0 2px'};
                      }
                      return null;
                    },
                    textStyle: const TextStyle(
                      fontSize: 40,
                      height: 2,
                      letterSpacing: 5,
                      color: Colors.black87,
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 24),
            Text(
              'Elapsed Time: $_elapsedSeconds s',
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 16),
            if (_isUploading)
              const Center(child: CircularProgressIndicator())
            else
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton(
                      onPressed: _isRecording ? null : _startRecording,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF1ABC9C),
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        textStyle: const TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                      child: const Text('Start'),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: ElevatedButton(
                      onPressed: _isRecording ? _stopRecording : null,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFFEF5350),
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        textStyle: const TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                      child: const Text('Stop'),
                    ),
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }
}