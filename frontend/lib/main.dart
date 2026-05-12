import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:io';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Kızılötesi Görüntü Anomali Tespiti',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: const AnomalyDetectionPage(),
    );
  }
}

class AnomalyDetectionPage extends StatefulWidget {
  const AnomalyDetectionPage({super.key});

  @override
  State<AnomalyDetectionPage> createState() => _AnomalyDetectionPageState();
}

class _AnomalyDetectionPageState extends State<AnomalyDetectionPage> {
  String? _selectedVideoPath;
  String? _selectedVideoName;
  bool _isProcessing = false;
  String? _jobId;

  // Results
  String? _resultLabel;
  double? _resultConfidence;
  int? _resultLatency;
  bool _hasResults = false;

  final String _backendUrl = 'http://localhost:8000';

  Future<void> _pickVideo() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.video,
      allowMultiple: false,
    );

    if (result != null && result.files.single.path != null) {
      setState(() {
        _selectedVideoPath = result.files.single.path;
        _selectedVideoName = result.files.single.name;
      });
    }
  }

  Future<void> _uploadVideo() async {
    if (_selectedVideoPath == null) {
      _showSnackBar('Lütfen bir video seçin', Colors.red);
      return;
    }

    setState(() {
      _isProcessing = true;
      _hasResults = false;
    });

    try {
      final file = File(_selectedVideoPath!);
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$_backendUrl/process-video'),
      );

      request.files.add(
        await http.MultipartFile.fromPath('file', file.path),
      );

      final response = await request.send();
      final _ = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        final jobId = 'job_${DateTime.now().millisecondsSinceEpoch}';

        setState(() {
          _jobId = jobId;
        });

        _showSnackBar('Video yüklendi! İşleniyor...', Colors.green);

        // Simulate processing with mock results
        await Future.delayed(const Duration(seconds: 2));
        _getMockResults();
      } else {
        _showSnackBar('Video yüklenemedi', Colors.red);
      }
    } catch (e) {
      _showSnackBar('Hata: $e', Colors.red);
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
  }

  void _getMockResults() {
    setState(() {
      _hasResults = true;
      _resultLabel = 'Normal';
      _resultConfidence = 0.87;
      _resultLatency = 1240; // ms
    });
  }

  void _showSnackBar(String message, Color color) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: color,
        duration: const Duration(seconds: 2),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Kızılötesi Görüntü Anomali Tespiti'),
        elevation: 0,
      ),
      body: Row(
        children: [
          // Left Panel - Video Upload
          Expanded(
            flex: 1,
            child: Container(
              color: Colors.grey[100],
              padding: const EdgeInsets.all(24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Video Yükle',
                    style: Theme.of(context).textTheme.headlineSmall,
                  ),
                  const SizedBox(height: 24),
                  // Upload Area
                  Container(
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.blue, width: 2),
                      borderRadius: BorderRadius.circular(12),
                      color: Colors.white,
                    ),
                    padding: const EdgeInsets.all(32),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(
                          Icons.video_library,
                          size: 64,
                          color: Colors.blue,
                        ),
                        const SizedBox(height: 16),
                        Text(
                          'Video Dosyası Seç',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'MP4, AVI, MOV formatları desteklenir',
                          style: Theme.of(context).textTheme.bodySmall,
                          textAlign: TextAlign.center,
                        ),
                        const SizedBox(height: 16),
                        ElevatedButton.icon(
                          onPressed: _isProcessing ? null : _pickVideo,
                          icon: const Icon(Icons.folder_open),
                          label: const Text('Dosya Seç'),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 24),
                  // Selected File Info
                  if (_selectedVideoName != null) ...[
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.green[50],
                        border: Border.all(color: Colors.green),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Seçilen Dosya:',
                            style: Theme.of(context).textTheme.labelSmall,
                          ),
                          const SizedBox(height: 4),
                          Text(
                            _selectedVideoName!,
                            style: Theme.of(context).textTheme.bodyMedium,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 16),
                  ],
                  // Upload Button
                  SizedBox(
                    width: double.infinity,
                    height: 48,
                    child: ElevatedButton.icon(
                      onPressed: _isProcessing ? null : _uploadVideo,
                      icon: _isProcessing
                          ? const SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                              ),
                            )
                          : const Icon(Icons.upload),
                      label: Text(
                        _isProcessing ? 'İşleniyor...' : 'Video İşle',
                        style: const TextStyle(fontSize: 16),
                      ),
                    ),
                  ),
                  const Spacer(),
                  // Info Box
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.blue[50],
                      border: Border.all(color: Colors.blue),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Sistem Durumu:',
                          style: Theme.of(context).textTheme.labelSmall,
                        ),
                        const SizedBox(height: 4),
                        Row(
                          children: [
                            const CircleAvatar(
                              radius: 4,
                              backgroundColor: Colors.green,
                            ),
                            const SizedBox(width: 8),
                            Text(
                              'Backend Bağlı',
                              style: Theme.of(context).textTheme.bodySmall,
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
          // Right Panel - Results
          Expanded(
            flex: 1,
            child: Container(
              color: Colors.white,
              padding: const EdgeInsets.all(24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Analiz Sonuçları',
                    style: Theme.of(context).textTheme.headlineSmall,
                  ),
                  const SizedBox(height: 24),
                  if (!_hasResults)
                    Expanded(
                      child: Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              Icons.hourglass_empty,
                              size: 64,
                              color: Colors.grey[400],
                            ),
                            const SizedBox(height: 16),
                            Text(
                              'Sonuç Bekleniyor',
                              style: Theme.of(context).textTheme.titleMedium,
                            ),
                            const SizedBox(height: 8),
                            Text(
                              'Video yükleyip işlemeye başlayın',
                              style: Theme.of(context).textTheme.bodySmall,
                              textAlign: TextAlign.center,
                            ),
                          ],
                        ),
                      ),
                    )
                  else
                    Expanded(
                      child: Column(
                        children: [
                          // Result Card
                          Container(
                            padding: const EdgeInsets.all(16),
                            decoration: BoxDecoration(
                              color: Colors.blue[50],
                              border: Border.all(color: Colors.blue),
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Column(
                              children: [
                                _resultItem(
                                  'Sınıf',
                                  _resultLabel ?? 'N/A',
                                  Icons.label,
                                ),
                                const SizedBox(height: 16),
                                _resultItem(
                                  'Güven Oranı',
                                  _resultConfidence != null
                                      ? '${(_resultConfidence! * 100).toStringAsFixed(1)}%'
                                      : 'N/A',
                                  Icons.percent,
                                ),
                                const SizedBox(height: 16),
                                _resultItem(
                                  'İşlem Süresi',
                                  _resultLatency != null
                                      ? '${_resultLatency}ms'
                                      : 'N/A',
                                  Icons.schedule,
                                ),
                              ],
                            ),
                          ),
                          const SizedBox(height: 24),
                          // Confidence Gauge
                          if (_resultConfidence != null)
                            Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'Güven Ölçeği',
                                  style: Theme.of(context).textTheme.labelMedium,
                                ),
                                const SizedBox(height: 8),
                                ClipRRect(
                                  borderRadius: BorderRadius.circular(8),
                                  child: LinearProgressIndicator(
                                    value: _resultConfidence!,
                                    minHeight: 12,
                                    backgroundColor: Colors.grey[300],
                                    valueColor: AlwaysStoppedAnimation(
                                      _resultConfidence! > 0.7
                                          ? Colors.green
                                          : _resultConfidence! > 0.5
                                              ? Colors.orange
                                              : Colors.red,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                        ],
                      ),
                    ),
                  const Spacer(),
                  // Action Buttons
                  if (_hasResults)
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton(
                        onPressed: () {
                          setState(() {
                            _hasResults = false;
                            _resultLabel = null;
                            _resultConfidence = null;
                            _resultLatency = null;
                          });
                        },
                        child: const Text('Yeni İşlem'),
                      ),
                    ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _resultItem(String label, String value, IconData icon) {
    return Row(
      children: [
        Icon(icon, color: Colors.blue),
        const SizedBox(width: 12),
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              label,
              style: Theme.of(context).textTheme.labelSmall,
            ),
            const SizedBox(height: 4),
            Text(
              value,
              style: Theme.of(context).textTheme.titleMedium,
            ),
          ],
        ),
      ],
    );
  }
}
