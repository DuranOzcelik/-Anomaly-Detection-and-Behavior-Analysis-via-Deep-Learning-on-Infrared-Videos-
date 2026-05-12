import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:io';
import 'services/api_service.dart';
import 'widgets/video_heatmap_overlay.dart';

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

  // Results
  String? _resultLabel;
  double? _resultConfidence;
  int? _resultLatency;
  String? _resultHeatmap;
  bool _hasResults = false;
  bool _showHeatmap = false;

  final ApiService _apiService = ApiService();

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
      _showHeatmap = false;
    });

    try {
      final file = File(_selectedVideoPath!);
      final response = await _apiService.uploadVideo(file);

      setState(() {
        _resultLabel = response.classification;
        _resultConfidence = response.confidence;
        _resultLatency = response.latencyMs;
        _resultHeatmap = response.heatmapBase64;
        _hasResults = true;
      });

      _showSnackBar('Video işlendi! Sonuçlar yüklendi.', Colors.green);
    } catch (e) {
      _showSnackBar('Hata: $e', Colors.red);
      setState(() {
        _hasResults = false;
      });
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
  }

  void _showSnackBar(String message, Color color) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: color,
        duration: const Duration(seconds: 3),
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
          // Right Panel - Results & Heatmap
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
                          const SizedBox(height: 24),
                          // Heatmap Button
                          if (_resultHeatmap != null)
                            SizedBox(
                              width: double.infinity,
                              child: ElevatedButton.icon(
                                onPressed: () {
                                  setState(() {
                                    _showHeatmap = !_showHeatmap;
                                  });
                                },
                                icon: Icon(_showHeatmap
                                    ? Icons.close
                                    : Icons.visibility),
                                label: Text(_showHeatmap
                                    ? 'Heatmap Kapat'
                                    : 'Heatmap Göster'),
                              ),
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
                            _resultHeatmap = null;
                            _showHeatmap = false;
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
      // Heatmap Modal
      floatingActionButton: _showHeatmap && _resultHeatmap != null
          ? FloatingActionButton(
              onPressed: () {
                setState(() {
                  _showHeatmap = false;
                });
              },
              child: const Icon(Icons.close),
            )
          : null,
    );
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // Heatmap gösteriliyorsa ve boyut değiştiyse, overlay'i güncelle
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
