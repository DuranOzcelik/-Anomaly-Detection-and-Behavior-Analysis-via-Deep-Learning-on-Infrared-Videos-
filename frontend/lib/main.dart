import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:video_player/video_player.dart';
import 'services/websocket_service.dart';
import 'services/api_service.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'IR Anomaly Detection',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: _blue,
          brightness: Brightness.light,
        ),
        useMaterial3: true,
        scaffoldBackgroundColor: _bg0,
      ),
      home: const AnomalyDetectionPage(),
    );
  }
}

// ─── Renk paleti ──────────────────────────────────────────────────────────────
const _bg0    = Color(0xFFF1F5F9);
const _bg1    = Color(0xFFFFFFFF);
const _bg2    = Color(0xFFEEF2F7);
const _bd     = Color(0xFFE2E8F0);
const _blue   = Color(0xFF2563EB);
const _violet = Color(0xFF7C3AED);
const _amber  = Color(0xFFD97706);
const _green  = Color(0xFF16A34A);
const _yellow = Color(0xFFCA8A04);
const _red    = Color(0xFFDC2626);
const _t0     = Color(0xFF0F172A);
const _t1     = Color(0xFF475569);
const _t2     = Color(0xFF64748B);

// ─── Ana sayfa ────────────────────────────────────────────────────────────────
class AnomalyDetectionPage extends StatefulWidget {
  const AnomalyDetectionPage({super.key});
  @override
  State<AnomalyDetectionPage> createState() => _AnomalyDetectionPageState();
}

// ─── Kamera per-clip bildirim modeli ─────────────────────────────────────────
class _ClipAlert {
  final String classification;
  final double mse;
  final double anomalyScore;
  final int    clipIdx;
  _ClipAlert({required this.classification, required this.mse,
              required this.anomalyScore,   required this.clipIdx});
}

class _AnomalyDetectionPageState extends State<AnomalyDetectionPage>
    with TickerProviderStateMixin {
  late WebSocketService _wsService;
  late ApiService _apiService;

  // ── Video analiz modu
  DateTime? _startDate;
  DateTime? _endDate;
  bool _isAnalyzing = false;

  // ── Kamera modu
  WebSocketService? _camWsService;
  bool _cameraMode      = false;
  bool _cameraConnected = false;
  String? _cameraFrameB64;
  int  _camClipIdx      = 0;
  int  _camAnomalyClips = 0;
  final List<_ClipAlert> _clipAlerts = [];

  // ── Drive izleme modu
  WebSocketService? _driveWsService;
  bool   _driveMode      = false;
  bool   _driveConnected = false;
  String? _driveStatusMsg;
  List<Map<String, dynamic>> _driveVideos     = [];
  Map<String, String>        _driveVideoStatus = {}; // filename → 'waiting'|'analyzing'|'complete'|'error'

  // Acil durdur için oturum genelinde birikim
  Map<String, int>  _accumulatedClassDist = {};
  final List<double> _sessionAnomalyScores = [];

  VideoPlayerController? _videoCtrl;
  bool _videoReady = false;
  bool _videoPaused = false;

  String? _videoName;
  int _frameNum = 0;
  int _clipIdx  = 0;
  int _totalClips = 0;

  double _rawMse       = 0.0;
  bool   _exceeded     = false;
  int    _craeMs       = 0;
  double _anomalyScore = 0.0;
  String? _origFrameB64;
  String? _reconFrameB64;

  final List<double> _mseHistory   = [];
  final List<bool>   _clipTimeline = [];
  static const int   _maxHistory   = 40;

  Map<String, double>  _classProbs    = {};
  List<List<dynamic>>  _top2          = [];
  String               _classification = 'N/A';
  double               _confidence    = 0.0;
  int                  _cnnMs         = 0;

  String? _heatmapB64;
  int     _heatmapMs = 0;
  int     _latencyMs = 0;

  int    _totalVideos    = 0;
  int    _processedVideos = 0;
  int    _anomalyClips   = 0;
  int    _anomalyVideos  = 0;
  int    _processSec     = 0;
  double _avgAnomaly     = 0.0;
  String? _dominantClass;

  bool _craeOpen    = true;
  bool _cnnOpen     = true;
  bool _gradcamOpen = true;

  static const double _threshold = 0.012;

  late AnimationController _pulseCtrl;
  late Animation<double>   _pulseAnim;

  @override
  void initState() {
    super.initState();
    _apiService = ApiService();
    _pulseCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    )..repeat(reverse: true);
    _pulseAnim = CurvedAnimation(parent: _pulseCtrl, curve: Curves.easeInOut);
    _initWs();
  }

  void _initWs() {
    _wsService = WebSocketService();
    _wsService.connect().then((_) {
      if (mounted) setState(() {});
      _wsService.messages.listen((m) {
        if (!mounted) return;
        setState(() => _onMessage(m));
      });
    }).catchError((e) {
      _snack('WS connection failed: $e', _red);
    });
  }

  void _onMessage(ProcessingMessage m) {
    switch (m.type) {
      case 'video_ready':
        if (m.videoUrl != null) _initVideo(m.videoUrl!);

      case 'video_start':
        _videoName = m.filename;
        _frameNum = 0; _clipIdx = 0; _totalClips = 0;
        _rawMse = 0; _exceeded = false;
        _classProbs = {}; _top2 = [];
        _heatmapB64 = null;
        _origFrameB64 = null; _reconFrameB64 = null;
        _mseHistory.clear(); _clipTimeline.clear();

      case 'frame_processed':
        _frameNum      = m.frameNumber       ?? _frameNum;
        _clipIdx       = m.clipIndex         ?? _clipIdx;
        _totalClips    = m.totalClips        ?? _totalClips;
        _anomalyScore  = m.anomalyScore      ?? _anomalyScore;
        _rawMse        = m.rawMse            ?? _rawMse;
        _exceeded      = m.thresholdExceeded ?? _exceeded;
        _craeMs        = m.craeTimeMs        ?? _craeMs;
        _classification = m.classification   ?? _classification;
        _confidence    = m.confidence        ?? _confidence;
        _cnnMs         = m.cnnTimeMs         ?? _cnnMs;
        _heatmapMs     = m.heatmapTimeMs     ?? _heatmapMs;
        _latencyMs     = m.latencyMs         ?? _latencyMs;
        _heatmapB64    = m.heatmapBase64     ?? _heatmapB64;
        _classProbs    = m.allClassProbs     ?? _classProbs;
        _top2          = m.top2Classes       ?? _top2;
        _avgAnomaly    = m.avgAnomalyScore   ?? _avgAnomaly;
        _origFrameB64  = m.origFrameBase64   ?? _origFrameB64;
        _reconFrameB64 = m.reconFrameBase64  ?? _reconFrameB64;

        _mseHistory.add(_rawMse);
        if (_mseHistory.length > _maxHistory) _mseHistory.removeAt(0);
        _clipTimeline.add(_exceeded);
        _sessionAnomalyScores.add(_anomalyScore);
        if (_exceeded) {
          _anomalyClips++;
          if (_classification != 'N/A') {
            _accumulatedClassDist[_classification] =
                (_accumulatedClassDist[_classification] ?? 0) + 1;
          }
        }

      case 'video_complete':
        _processedVideos++;

      case 'video_summary':
        _avgAnomaly    = m.avgAnomalyScore ?? _avgAnomaly;
        _dominantClass = m.dominantClass   ?? _dominantClass;
        final snapFilename   = _videoName ?? 'Unknown';
        final snapTotal      = _clipTimeline.length;
        final snapAnomalies  = _anomalyClips;
        final snapAvg        = _avgAnomaly;
        final snapDominant   = _dominantClass ?? 'Normal';
        final snapDist       = Map<String, int>.from(
            m.classDistribution ?? <String, int>{});
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (mounted) {
            _showVideoSummaryAlert(
              filename:   snapFilename,
              totalClips: snapTotal,
              anomalies:  snapAnomalies,
              avgAnomaly: snapAvg,
              dominant:   snapDominant,
              classDist:  snapDist,
            );
          }
        });

      case 'batch_complete':
        _isAnalyzing = false;
        _totalVideos = m.totalVideos ?? _totalVideos;
        _processSec  = m.processingTimeSeconds ?? _processSec;
        _snack('Analysis complete! $_processedVideos videos, $_anomalyClips anomaly clips.', _green);

      case 'batch_summary':
        _anomalyVideos   = m.totalVideosWithAnomalies ?? _anomalyVideos;
        _processSec      = m.processingTimeSeconds    ?? _processSec;
        _avgAnomaly      = m.avgAnomalyAcrossAll      ?? _avgAnomaly;
        _processedVideos = m.totalVideos              ?? _processedVideos;

      case 'error':
        _isAnalyzing = false;
        _snack('Error: ${m.errorMessage}', _red);
    }
  }

  Future<void> _initVideo(String url) async {
    await _videoCtrl?.dispose();
    _videoCtrl = null;
    _videoReady = false;
    _videoPaused = false;
    try {
      final c = VideoPlayerController.networkUrl(Uri.parse(url));
      _videoCtrl = c;
      await c.initialize();
      c.setLooping(false);
      c.play();
      if (mounted) setState(() => _videoReady = true);
    } catch (e) {
      debugPrint('Video init error: $e');
    }
  }

  void _toggleVideo() {
    final c = _videoCtrl;
    if (c == null) return;
    setState(() {
      if (_videoPaused) { c.play(); _videoPaused = false; }
      else              { c.pause(); _videoPaused = true; }
    });
  }

  Future<void> _pickStartDate() async {
    final d = await showDatePicker(
      context: context,
      initialDate: _startDate ?? DateTime.now(),
      firstDate: DateTime(2020), lastDate: DateTime.now(),
      builder: (ctx, child) =>
          Theme(data: ThemeData.light(useMaterial3: true), child: child!),
    );
    if (d != null && mounted) setState(() => _startDate = d);
  }

  Future<void> _pickEndDate() async {
    final d = await showDatePicker(
      context: context,
      initialDate: _endDate ?? (_startDate ?? DateTime.now()),
      firstDate: _startDate ?? DateTime(2020), lastDate: DateTime.now(),
      builder: (ctx, child) =>
          Theme(data: ThemeData.light(useMaterial3: true), child: child!),
    );
    if (d != null && mounted) setState(() => _endDate = d);
  }

  Future<void> _startAnalysis() async {
    if (_startDate == null || _endDate == null) {
      _snack('Please select a date range', _yellow); return;
    }
    if (!_wsService.isConnected) {
      _snack('WebSocket not connected', _red); return;
    }
    setState(() {
      _isAnalyzing = true; _processedVideos = 0; _anomalyClips = 0;
      _heatmapB64 = null; _videoName = null;
      _accumulatedClassDist.clear(); _sessionAnomalyScores.clear();
    });
    try {
      await _wsService.startAnalysis(
        startDate: DateFormat('yyyyMMdd').format(_startDate!),
        endDate: DateFormat('yyyyMMdd').format(_endDate!),
      );
    } catch (e) {
      if (mounted) setState(() => _isAnalyzing = false);
      _snack('Failed to start analysis: $e', _red);
    }
  }

  Future<void> _openVideoList() async {
    if (_startDate == null || _endDate == null) {
      _snack('Please select dates first', _yellow); return;
    }
    try {
      final videos = await _apiService.listVideos(
        DateFormat('yyyyMMdd').format(_startDate!),
        DateFormat('yyyyMMdd').format(_endDate!),
      );
      if (!mounted) return;
      if (videos.isEmpty) { _snack('No videos found', _yellow); return; }

      final sel = await showDialog<String>(
        context: context,
        builder: (_) => _VideoListDialog(
        videos: videos.cast<Map<String, dynamic>>()),
      );
      if (sel != null) {
        setState(() {
          _videoName = sel; _isAnalyzing = true;
          _processedVideos = 0; _anomalyClips = 0; _heatmapB64 = null;
        });
        await _wsService.startVideoProcessing(filename: sel);
      }
    } catch (e) {
      _snack('Failed to load video list: $e', _red);
    }
  }

  void _snack(String msg, Color c) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg), backgroundColor: c,
        duration: const Duration(seconds: 4)));
  }

  // ─── Kamera modu ────────────────────────────────────────────────────────────

  Future<void> _startCamera() async {
    setState(() {
      _cameraMode      = true;
      _cameraConnected = false;
      _cameraFrameB64  = null;
      _camClipIdx      = 0;
      _camAnomalyClips = 0;
      _clipAlerts.clear();
      _rawMse = 0; _exceeded = false; _mseHistory.clear();
      _clipTimeline.clear(); _heatmapB64 = null;
      _origFrameB64 = null; _reconFrameB64 = null;
      _classProbs = {}; _top2 = []; _classification = 'N/A';
      _accumulatedClassDist.clear(); _sessionAnomalyScores.clear();
    });

    try {
      _camWsService = WebSocketService(path: '/ws/camera');
      await _camWsService!.connect();
      _camWsService!.messages.listen((m) {
        if (!mounted) return;
        setState(() => _onCameraMessage(m));
      });
      await _camWsService!.startCamera();
    } catch (e) {
      setState(() => _cameraMode = false);
      _snack('Camera connection failed: $e', _red);
    }
  }

  Future<void> _stopCamera() async {
    try {
      await _camWsService?.stopCamera();
    } catch (_) {}
    _camWsService?.dispose();
    _camWsService = null;
    setState(() {
      _cameraMode      = false;
      _cameraConnected = false;
      _cameraFrameB64  = null;
      _clipAlerts.clear();
    });
  }

  void _onCameraMessage(ProcessingMessage m) {
    switch (m.type) {
      case 'camera_connected':
        _cameraConnected = true;

      case 'camera_frame':
        _cameraFrameB64 = m.cameraFrameBase64 ?? _cameraFrameB64;

      case 'clip_result':
        // Pipeline panellerini güncelle (frame_processed ile aynı mantık)
        _frameNum      = m.frameNumber       ?? _frameNum;
        _camClipIdx    = (m.clipIndex ?? _camClipIdx - 1) + 1;
        _anomalyScore  = m.anomalyScore      ?? _anomalyScore;
        _rawMse        = m.rawMse            ?? _rawMse;
        _exceeded      = m.thresholdExceeded ?? _exceeded;
        _craeMs        = m.craeTimeMs        ?? _craeMs;
        _classification = m.classification   ?? _classification;
        _confidence    = m.confidence        ?? _confidence;
        _cnnMs         = m.cnnTimeMs         ?? _cnnMs;
        _heatmapMs     = m.heatmapTimeMs     ?? _heatmapMs;
        _latencyMs     = m.latencyMs         ?? _latencyMs;
        _heatmapB64    = m.heatmapBase64     ?? _heatmapB64;
        _classProbs    = m.allClassProbs     ?? _classProbs;
        _top2          = m.top2Classes       ?? _top2;
        _origFrameB64  = m.origFrameBase64   ?? _origFrameB64;
        _reconFrameB64 = m.reconFrameBase64  ?? _reconFrameB64;

        _mseHistory.add(_rawMse);
        if (_mseHistory.length > _maxHistory) _mseHistory.removeAt(0);
        _clipTimeline.add(_exceeded);
        _sessionAnomalyScores.add(_anomalyScore);
        if (_exceeded && _classification != 'N/A') {
          _accumulatedClassDist[_classification] =
              (_accumulatedClassDist[_classification] ?? 0) + 1;
        }

        // Sadece anomali varsa küçük bildirim ekle
        if (_exceeded) {
          _camAnomalyClips++;
          _clipAlerts.add(_ClipAlert(
            classification: _classification,
            mse:            _rawMse,
            anomalyScore:   _anomalyScore,
            clipIdx:        _camClipIdx,
          ));
          // 4 saniye sonra bildirimi kaldır
          Future.delayed(const Duration(seconds: 4), () {
            if (mounted) {
              setState(() {
                if (_clipAlerts.isNotEmpty) _clipAlerts.removeAt(0);
              });
            }
          });
        }

      case 'camera_summary':
        // Sadece anomali varsa özet dialog göster
        if (_camAnomalyClips > 0) {
          final snapTotal    = _camClipIdx;
          final snapAnom     = _camAnomalyClips;
          final snapAvg      = m.avgAnomalyScore  ?? 0.0;
          final snapDominant = m.dominantClass    ?? 'Normal';
          final snapDist     = Map<String, int>.from(
              m.classDistribution ?? <String, int>{});
          WidgetsBinding.instance.addPostFrameCallback((_) {
            if (mounted) {
              _showVideoSummaryAlert(
                filename:   'Live Camera Feed',
                totalClips: snapTotal,
                anomalies:  snapAnom,
                avgAnomaly: snapAvg,
                dominant:   snapDominant,
                classDist:  snapDist,
              );
            }
          });
        }

      case 'camera_stopped':
        _cameraMode      = false;
        _cameraConnected = false;

      case 'error':
        _cameraMode = false;
        _snack('Camera error: ${m.errorMessage}', _red);
    }
  }

  // ─── Drive izleme modu ──────────────────────────────────────────────────────

  // Drive bağlanma dialogunu aç: klasör bilgisi + video listesi göster
  Future<void> _openDriveDialog() async {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => const _LoadingDialog(message: 'Loading Drive folder...'),
    );
    try {
      final data = await _apiService.listDriveVideos();
      if (!mounted) return;
      Navigator.of(context, rootNavigator: true).pop();

      final folderPath = data['folder_path'] as String;
      final videos = (data['videos'] as List).cast<Map<String, dynamic>>();

      if (videos.isEmpty) {
        _snack('No videos found in folder: $folderPath', _yellow);
        return;
      }

      await showDialog(
        context: context,
        builder: (_) => _DriveSelectDialog(
          folderPath: folderPath,
          videos: videos,
          onSelectAll: () {
            Navigator.of(context, rootNavigator: true).pop();
            _startDriveWatch(videos: videos);
          },
          onSelectVideo: (fileId, filename) {
            Navigator.of(context, rootNavigator: true).pop();
            _startDriveWatch(videos: videos, fileId: fileId, filename: filename);
          },
        ),
      );
    } catch (e) {
      if (mounted) {
        Navigator.of(context, rootNavigator: true).pop();
        _snack('Failed to load Drive folder: $e', _red);
      }
    }
  }

  Future<void> _startDriveWatch({
    List<Map<String, dynamic>>? videos,
    String? fileId,
    String? filename,
  }) async {
    setState(() {
      _driveMode      = true;
      _driveConnected = false;
      _driveStatusMsg = 'Connecting...';
      _driveVideos    = videos ?? [];
      _driveVideoStatus = {
        for (final v in (videos ?? [])) (v['filename'] as String): 'waiting',
      };
      _rawMse = 0; _exceeded = false; _mseHistory.clear();
      _clipTimeline.clear(); _heatmapB64 = null;
      _origFrameB64 = null; _reconFrameB64 = null;
      _classProbs = {}; _top2 = []; _classification = 'N/A';
      _processedVideos = 0; _anomalyClips = 0; _videoName = null;
      _accumulatedClassDist.clear(); _sessionAnomalyScores.clear();
    });

    try {
      _driveWsService = WebSocketService(path: '/ws/drive');
      await _driveWsService!.connect();
      _driveWsService!.messages.listen((m) {
        if (!mounted) return;
        setState(() => _onDriveMessage(m));
      });
      await _driveWsService!.startDriveWatch(fileId: fileId, filename: filename);
    } catch (e) {
      setState(() { _driveMode = false; _driveStatusMsg = null; });
      _snack('Drive connection failed: $e', _red);
    }
  }

  // Sol panelden farklı bir videoya geç (mevcut WS kesilir, yeni bağlantı açılır)
  Future<void> _switchDriveVideo(String fileId, String filename) async {
    try { await _driveWsService?.stopDriveWatch(); } catch (_) {}
    _driveWsService?.dispose();
    _driveWsService = null;
    setState(() {
      _driveConnected = false;
      _driveStatusMsg = 'Connecting...';
      _rawMse = 0; _exceeded = false; _mseHistory.clear();
      _clipTimeline.clear(); _heatmapB64 = null;
      _origFrameB64 = null; _reconFrameB64 = null;
      _classProbs = {}; _top2 = []; _classification = 'N/A';
      _videoName = null;
    });
    try {
      _driveWsService = WebSocketService(path: '/ws/drive');
      await _driveWsService!.connect();
      _driveWsService!.messages.listen((m) {
        if (!mounted) return;
        setState(() => _onDriveMessage(m));
      });
      await _driveWsService!.startDriveWatch(fileId: fileId, filename: filename);
    } catch (e) {
      setState(() { _driveMode = false; _driveStatusMsg = null; });
      _snack('Drive connection failed: $e', _red);
    }
  }

  // Analizi acil durdur + o ana kadar ki sonuçları popup olarak göster
  Future<void> _emergencyStop() async {
    // Mevcut oturum verisini yakala
    final hasData = _clipTimeline.isNotEmpty;
    final snapFilename = _videoName ??
        (_cameraMode ? 'Live Camera Feed' :
         _driveMode  ? 'Drive Integration Test' : 'Video Analysis');
    final snapTotal   = _clipTimeline.length;
    final snapAnom    = _anomalyClips;
    final snapAvg     = _sessionAnomalyScores.isNotEmpty
        ? _sessionAnomalyScores.reduce((a, b) => a + b) / _sessionAnomalyScores.length
        : _avgAnomaly;
    final snapDom     = _dominantClass ?? (_classification != 'N/A' ? _classification : 'Normal');
    final snapDist    = Map<String, int>.from(_accumulatedClassDist);

    // Analizi durdur
    if (_cameraMode) {
      await _stopCamera();
    } else if (_driveMode) {
      await _stopDriveWatch();
    } else if (_isAnalyzing) {
      try { await _wsService.stopAnalysis(); } catch (_) {}
      if (mounted) setState(() => _isAnalyzing = false);
    }

    // Popup göster
    if (hasData && mounted) {
      _showVideoSummaryAlert(
        filename:   snapFilename,
        totalClips: snapTotal,
        anomalies:  snapAnom,
        avgAnomaly: snapAvg,
        dominant:   snapDom,
        classDist:  snapDist,
      );
    }
  }

  Future<void> _stopDriveWatch() async {
    try { await _driveWsService?.stopDriveWatch(); } catch (_) {}
    _driveWsService?.dispose();
    _driveWsService = null;
    setState(() {
      _driveMode      = false;
      _driveConnected = false;
      _driveStatusMsg = null;
    });
  }

  void _onDriveMessage(ProcessingMessage m) {
    switch (m.type) {
      case 'drive_connected':
        _driveConnected = true;
        _driveStatusMsg = 'Folder connected';
      case 'drive_status':
        _driveStatusMsg = m.message ?? _driveStatusMsg;
      case 'video_start':
        if (m.filename != null) _driveVideoStatus[m.filename!] = 'analyzing';
        _onMessage(m);
      case 'video_complete':
        if (m.filename != null) _driveVideoStatus[m.filename!] = 'complete';
        _onMessage(m);
      case 'drive_poll_error':
        _snack('Drive poll error: ${m.errorMessage}', _yellow);
      case 'drive_stopped':
        _driveMode      = false;
        _driveConnected = false;
        _driveStatusMsg = null;
      case 'error':
        if (m.filename != null) _driveVideoStatus[m.filename!] = 'error';
        _driveMode      = false;
        _driveConnected = false;
        _driveStatusMsg = null;
        _snack('Drive error: ${m.errorMessage}', _red);
      default:
        _onMessage(m);
    }
  }

  // Herhangi bir widget'a "tıkla → tam ekran" davranışı ekler
  Widget _tapToExpand(Widget child, {Widget? expanded, Color bgColor = Colors.black}) {
    return GestureDetector(
      onTap: () => showDialog(
        context: context,
        barrierColor: Colors.black87,
        builder: (_) => Dialog(
          backgroundColor: bgColor,
          insetPadding: const EdgeInsets.all(12),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          child: Stack(children: [
            Center(child: Padding(
              padding: const EdgeInsets.all(12),
              child: expanded ?? child,
            )),
            Positioned(
              top: 8, right: 8,
              child: GestureDetector(
                onTap: () => Navigator.of(context, rootNavigator: true).pop(),
                child: Container(
                  padding: const EdgeInsets.all(5),
                  decoration: const BoxDecoration(
                    color: Color(0xAA000000), shape: BoxShape.circle),
                  child: const Icon(Icons.close, color: Colors.white70, size: 16),
                ),
              ),
            ),
          ]),
        ),
      ),
      child: child,
    );
  }

  void _showVideoSummaryAlert({
    required String filename,
    required int totalClips,
    required int anomalies,
    required double avgAnomaly,
    required String dominant,
    required Map<String, int> classDist,
  }) {
    final anomalyRatio = totalClips > 0 ? anomalies / totalClips : 0.0;
    final pct          = (avgAnomaly * 100).toStringAsFixed(1);

    // Risk seviyesi
    final String riskLabel;
    final Color  riskColor;
    final IconData riskIcon;
    final String riskNote;
    if (avgAnomaly >= 0.45 || anomalyRatio >= 0.6) {
      riskLabel = 'CRITICAL';
      riskColor = _red;
      riskIcon  = Icons.warning_rounded;
      riskNote  = 'Immediate on-site inspection recommended. Notify authorized personnel.';
    } else if (avgAnomaly >= 0.2 || anomalyRatio >= 0.25) {
      riskLabel = 'MODERATE';
      riskColor = _yellow;
      riskIcon  = Icons.info_rounded;
      riskNote  = 'Deviation detected. Review the records for further investigation.';
    } else {
      riskLabel = 'NORMAL';
      riskColor = _green;
      riskIcon  = Icons.check_circle_rounded;
      riskNote  = 'No significant anomalies detected. Routine monitoring is sufficient.';
    }

    final now = DateFormat('HH:mm  dd.MM.yyyy').format(DateTime.now());

    showDialog(
      context: context,
      barrierColor: Colors.black.withValues(alpha: 0.65),
      builder: (_) => Dialog(
        backgroundColor: Colors.transparent,
        insetPadding: const EdgeInsets.symmetric(horizontal: 32, vertical: 40),
        child: Container(
          constraints: const BoxConstraints(maxWidth: 480),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: riskColor.withValues(alpha: 0.45), width: 1.5),
            boxShadow: [
              BoxShadow(
                color: riskColor.withValues(alpha: 0.18),
                blurRadius: 32, spreadRadius: 2),
            ],
          ),
          child: Column(mainAxisSize: MainAxisSize.min, children: [

            // ── Başlık
            Container(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 14),
              decoration: BoxDecoration(
                color: riskColor.withValues(alpha: 0.08),
                borderRadius: const BorderRadius.vertical(top: Radius.circular(15)),
                border: Border(bottom: BorderSide(color: riskColor.withValues(alpha: 0.2))),
              ),
              child: Row(children: [
                Icon(riskIcon, color: riskColor, size: 20),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    const Text('ANALYSIS REPORT',
                      style: TextStyle(color: _t0, fontSize: 13,
                        fontWeight: FontWeight.w700, letterSpacing: 0.8)),
                    Text(filename,
                      style: const TextStyle(color: _t1, fontSize: 11),
                      overflow: TextOverflow.ellipsis),
                  ]),
                ),
                Text(now, style: const TextStyle(color: _t2, fontSize: 10)),
              ]),
            ),

            Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 6),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [

                // ── Risk seviyesi rozeti
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                  decoration: BoxDecoration(
                    color: riskColor.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: riskColor.withValues(alpha: 0.35)),
                  ),
                  child: Row(children: [
                    Text('RISK LEVEL',
                      style: TextStyle(color: riskColor.withValues(alpha: 0.7),
                        fontSize: 10, letterSpacing: 1.2)),
                    const SizedBox(width: 12),
                    Text(riskLabel,
                      style: TextStyle(color: riskColor,
                        fontSize: 18, fontWeight: FontWeight.w800, letterSpacing: 1)),
                    const Spacer(),
                    // Anomali oranı bar
                    Column(crossAxisAlignment: CrossAxisAlignment.end, children: [
                      Text('${(anomalyRatio * 100).toStringAsFixed(0)}% anomaly clips',
                        style: TextStyle(color: riskColor, fontSize: 10,
                          fontWeight: FontWeight.w600)),
                      const SizedBox(height: 4),
                      SizedBox(
                        width: 90,
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(3),
                          child: LinearProgressIndicator(
                            value: anomalyRatio.clamp(0.0, 1.0),
                            minHeight: 5,
                            backgroundColor: Colors.black12,
                            valueColor: AlwaysStoppedAnimation(riskColor),
                          ),
                        ),
                      ),
                    ]),
                  ]),
                ),

                const SizedBox(height: 14),

                // ── Metrik özeti
                _alertSectionLabel('DETECTION SUMMARY'),
                const SizedBox(height: 8),
                Row(children: [
                  Expanded(child: _alertMetric('Total Clips',   '$totalClips',  _t0)),
                  const SizedBox(width: 8),
                  Expanded(child: _alertMetric('Anomaly Clips', '$anomalies',   riskColor)),
                  const SizedBox(width: 8),
                  Expanded(child: _alertMetric('Avg. Anomaly',  '$pct%',        _yellow)),
                ]),
                const SizedBox(height: 8),
                _alertMetricWide('Dominant Class', dominant, _classColor(dominant)),

                // ── Clip distribution
                const SizedBox(height: 12),
                _alertSectionLabel('CLIP DISTRIBUTION'),
                const SizedBox(height: 6),
                _alertDistRow('Normal',  totalClips - anomalies, totalClips, _green),
                const SizedBox(height: 4),
                _alertDistRow('Anomaly', anomalies, totalClips, riskColor),

                if (classDist.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  _alertSectionLabel('CLASS DISTRIBUTION  (anomaly clips)'),
                  const SizedBox(height: 6),
                  ...classDist.entries.map((e) {
                    // Sınıf dağılımı anomali kliplere oranlanır
                    final base = anomalies > 0 ? anomalies : 1;
                    final ratio = e.value / base;
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 5),
                      child: Row(children: [
                        SizedBox(width: 110,
                          child: Text(e.key,
                            style: TextStyle(
                              color: _classColor(e.key), fontSize: 10,
                              fontWeight: FontWeight.w500),
                            overflow: TextOverflow.ellipsis)),
                        Expanded(
                          child: ClipRRect(
                            borderRadius: BorderRadius.circular(2),
                            child: LinearProgressIndicator(
                              value: ratio.clamp(0.0, 1.0),
                              minHeight: 5,
                              backgroundColor: Colors.black12,
                              valueColor: AlwaysStoppedAnimation(
                                _classColor(e.key).withValues(alpha: 0.8)),
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                        SizedBox(width: 32,
                          child: Text('${e.value}',
                            textAlign: TextAlign.right,
                            style: const TextStyle(color: _t1, fontSize: 10,
                              fontWeight: FontWeight.w600))),
                      ]),
                    );
                  }),
                ],

                const SizedBox(height: 12),

                // ── Operatör notu
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: _bg2,
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: _bd),
                  ),
                  child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Icon(Icons.person_outlined, size: 14, color: _t2),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(riskNote,
                        style: const TextStyle(color: _t1, fontSize: 11, height: 1.5)),
                    ),
                  ]),
                ),

                const SizedBox(height: 16),

                // ── Butonlar
                Row(children: [
                  Expanded(
                    child: TextButton(
                      onPressed: () => Navigator.of(context, rootNavigator: true).pop(),
                      style: TextButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        foregroundColor: _t1,
                        backgroundColor: _bg2,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                          side: const BorderSide(color: _bd)),
                      ),
                      child: const Text('CLOSE', style: TextStyle(fontSize: 12,
                        letterSpacing: 0.8, fontWeight: FontWeight.w600)),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: ElevatedButton(
                      onPressed: () {
                        Navigator.of(context, rootNavigator: true).pop();
                        setState(() {
                          _isAnalyzing = false;
                          _processedVideos = 0;
                          _anomalyClips = 0;
                        });
                      },
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        backgroundColor: _blue,
                        foregroundColor: Colors.white,
                        elevation: 0,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8)),
                      ),
                      child: const Text('NEW ANALYSIS', style: TextStyle(fontSize: 12,
                        letterSpacing: 0.8, fontWeight: FontWeight.w600)),
                    ),
                  ),
                ]),

                const SizedBox(height: 4),
              ]),
            ),
          ]),
        ),
      ),
    );
  }

  Widget _alertDistRow(String label, int count, int total, Color color) {
    final ratio = total > 0 ? count / total : 0.0;
    final pct   = (ratio * 100).toStringAsFixed(0);
    return Row(children: [
      SizedBox(width: 58,
        child: Text(label,
          style: TextStyle(color: color, fontSize: 10, fontWeight: FontWeight.w500))),
      Expanded(
        child: ClipRRect(
          borderRadius: BorderRadius.circular(2),
          child: LinearProgressIndicator(
            value: ratio.clamp(0.0, 1.0),
            minHeight: 6,
            backgroundColor: Colors.black12,
            valueColor: AlwaysStoppedAnimation(color.withValues(alpha: 0.8)),
          ),
        ),
      ),
      const SizedBox(width: 8),
      SizedBox(width: 42,
        child: Text('$count  ($pct%)',
          textAlign: TextAlign.right,
          style: TextStyle(color: color, fontSize: 10, fontWeight: FontWeight.w600))),
    ]);
  }

  Widget _alertSectionLabel(String text) => Padding(
    padding: const EdgeInsets.only(bottom: 2),
    child: Text(text,
      style: const TextStyle(color: _t2, fontSize: 9, letterSpacing: 1.3)));

  Widget _alertMetric(String label, String value, Color valueColor) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: _bg2, borderRadius: BorderRadius.circular(8),
        border: Border.all(color: _bd),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(label, style: const TextStyle(color: _t2, fontSize: 9)),
        const SizedBox(height: 3),
        Text(value, style: TextStyle(color: valueColor, fontSize: 15,
          fontWeight: FontWeight.w700)),
      ]),
    );
  }

  Widget _alertMetricWide(String label, String value, Color valueColor) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: _bg2, borderRadius: BorderRadius.circular(8),
        border: Border.all(color: _bd),
      ),
      child: Row(children: [
        Text(label, style: const TextStyle(color: _t2, fontSize: 10)),
        const SizedBox(width: 10),
        Text(value, style: TextStyle(color: valueColor, fontSize: 12,
          fontWeight: FontWeight.w700), overflow: TextOverflow.ellipsis),
      ]),
    );
  }

  Uint8List _decode(String b64) {
    final s = b64.contains(',') ? b64.split(',').last : b64;
    try { return base64Decode(s); } catch (_) { return Uint8List(0); }
  }

  @override
  void dispose() {
    _pulseCtrl.dispose();
    _videoCtrl?.dispose();
    _wsService.dispose();
    _camWsService?.dispose();
    _driveWsService?.dispose();
    super.dispose();
  }

  // ─── Build ──────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _bg0,
      appBar: _appBar(),
      body: Row(children: [_leftPanel(), Expanded(child: _rightPanel())]),
    );
  }

  // ─── AppBar ─────────────────────────────────────────────────────────────────

  AppBar _appBar() {
    final connected = _wsService.isConnected;
    return AppBar(
      backgroundColor: _bg1,
      elevation: 0,
      surfaceTintColor: Colors.transparent,
      titleSpacing: 20,
      bottom: PreferredSize(
        preferredSize: const Size.fromHeight(1),
        child: Container(color: _bd, height: 1),
      ),
      title: Row(children: [
        Container(width: 7, height: 7,
          decoration: const BoxDecoration(color: _blue, shape: BoxShape.circle)),
        const SizedBox(width: 10),
        const Text('IR Anomaly Detection',
          style: TextStyle(color: _t0, fontSize: 15, fontWeight: FontWeight.w600,
            letterSpacing: 0.3)),
        const SizedBox(width: 8),
        const Text('v2.1',
          style: TextStyle(color: _t2, fontSize: 11)),
      ]),
      actions: [
        if (_isAnalyzing || _cameraMode || _driveMode)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 16),
            child: Row(children: [
              FadeTransition(
                opacity: _pulseAnim,
                child: Icon(Icons.circle, size: 8,
                  color: _driveMode ? _blue : (_cameraMode ? const Color(0xFF0D9488) : _red))),
              const SizedBox(width: 6),
              Text(
                _driveMode ? 'DRIVE' : (_cameraMode ? 'CAMERA' : 'LIVE'),
                style: TextStyle(
                  color: _driveMode ? _blue : (_cameraMode ? const Color(0xFF0D9488) : _red),
                  fontSize: 10, letterSpacing: 2, fontWeight: FontWeight.w600)),
            ]),
          ),
        Container(
          margin: const EdgeInsets.symmetric(horizontal: 14, vertical: 11),
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
          decoration: BoxDecoration(
            color: connected ? _green.withValues(alpha: 0.08) : _red.withValues(alpha: 0.08),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: connected
                  ? _green.withValues(alpha: 0.45)
                  : _red.withValues(alpha: 0.45)),
          ),
          child: Row(children: [
            Container(width: 6, height: 6,
              decoration: BoxDecoration(
                color: connected ? _green : _red, shape: BoxShape.circle)),
            const SizedBox(width: 6),
            Text(connected ? 'Backend Connected' : 'Disconnected',
              style: TextStyle(color: connected ? _green : _red,
                fontSize: 11, fontWeight: FontWeight.w500)),
          ]),
        ),
      ],
    );
  }

  // ─── Sol panel ──────────────────────────────────────────────────────────────

  Widget _leftPanel() {
    final busy = _isAnalyzing || _cameraMode || _driveMode;
    return Container(
      width: 260,
      decoration: const BoxDecoration(
        color: _bg1,
        border: Border(right: BorderSide(color: _bd)),
      ),
      padding: const EdgeInsets.all(14),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [

        // ── Video modu
        _label('Video Analysis'),
        const SizedBox(height: 8),
        _sideBtn(Icons.videocam_outlined, 'Select Video',
          onTap: busy ? null : _openVideoList, color: _blue),
        const SizedBox(height: 8),
        _dateBtn(
          _startDate != null
              ? DateFormat('dd.MM.yyyy').format(_startDate!)
              : 'Start Date',
          onTap: busy ? null : _pickStartDate,
        ),
        const SizedBox(height: 5),
        _dateBtn(
          _endDate != null
              ? DateFormat('dd.MM.yyyy').format(_endDate!)
              : 'End Date',
          onTap: busy ? null : _pickEndDate,
        ),
        const SizedBox(height: 8),
        _sideBtn(
          _isAnalyzing ? Icons.hourglass_top : Icons.play_arrow_rounded,
          _isAnalyzing ? 'Processing...' : 'Start Analysis',
          onTap: busy ? null : _startAnalysis,
          color: const Color(0xFF1D4ED8),
          loading: _isAnalyzing,
        ),

        const SizedBox(height: 14),
        const Divider(color: _bd, height: 1),
        const SizedBox(height: 14),

        // ── Kamera modu
        _label('Live Camera'),
        const SizedBox(height: 8),
        _sideBtn(
          _cameraMode ? Icons.stop_circle_outlined : Icons.videocam_rounded,
          _cameraMode
              ? (_cameraConnected ? 'Stop Camera' : 'Connecting...')
              : 'Connect Camera',
          onTap: busy && !_cameraMode
              ? null
              : (_cameraMode ? _stopCamera : _startCamera),
          color: _cameraMode ? _red : const Color(0xFF0D7377),
          loading: _cameraMode && !_cameraConnected,
        ),
        if (_cameraMode && _cameraConnected) ...[
          const SizedBox(height: 6),
          _camStatusRow(),
        ],

        const SizedBox(height: 14),
        const Divider(color: _bd, height: 1),
        const SizedBox(height: 14),

        // ── Integration Test
        _label('Integration Test'),
        const SizedBox(height: 8),
        _sideBtn(
          _driveMode ? Icons.stop_circle_outlined : Icons.science_outlined,
          _driveMode
              ? (_driveConnected ? 'Stop Test' : 'Connecting...')
              : 'Start System Test',
          onTap: busy && !_driveMode
              ? null
              : (_driveMode ? _stopDriveWatch : _openDriveDialog),
          color: _driveMode ? _red : const Color(0xFF1565C0),
          loading: _driveMode && !_driveConnected,
        ),
        if (_driveMode && _driveConnected) ...[
          const SizedBox(height: 6),
          _driveStatusRow(),
        ],
        if (_driveMode && _driveVideos.isNotEmpty) ...[
          const SizedBox(height: 6),
          _driveVideoListWidget(),
        ],

        const SizedBox(height: 20),
        _label('Statistics'),
        const SizedBox(height: 8),
        _statsPanel(),
        if (_isAnalyzing || _cameraMode || _driveMode) ...[
          const SizedBox(height: 12),
          _stopAnalysisBtn(),
        ],
        const Spacer(),
        if (_latencyMs > 0) _timingCard(),
      ]),
    );
  }

  Widget _camStatusRow() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: _bg1, borderRadius: BorderRadius.circular(7),
        border: Border.all(color: const Color(0xFF0D7377).withValues(alpha: 0.4)),
      ),
      child: Column(children: [
        Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
          const Text('Clip', style: TextStyle(color: _t1, fontSize: 10)),
          Text('$_camClipIdx',
            style: const TextStyle(color: _t0, fontSize: 10, fontWeight: FontWeight.w600)),
        ]),
        const SizedBox(height: 3),
        Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
          const Text('Anomalies', style: TextStyle(color: _t1, fontSize: 10)),
          Text('$_camAnomalyClips',
            style: TextStyle(
              color: _camAnomalyClips > 0 ? _red : _green,
              fontSize: 10, fontWeight: FontWeight.w600)),
        ]),
      ]),
    );
  }

  Widget _driveStatusRow() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: _bg1, borderRadius: BorderRadius.circular(7),
        border: Border.all(color: const Color(0xFF1565C0).withValues(alpha: 0.4)),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
          const Text('Drive', style: TextStyle(color: _t1, fontSize: 10)),
          const Icon(Icons.cloud_sync_outlined, size: 12, color: _blue),
        ]),
        if (_driveStatusMsg != null) ...[
          const SizedBox(height: 3),
          Text(_driveStatusMsg!,
            style: const TextStyle(color: _t0, fontSize: 9),
            overflow: TextOverflow.ellipsis),
        ],
        const SizedBox(height: 3),
        Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
          const Text('Anomaly Clips', style: TextStyle(color: _t1, fontSize: 10)),
          Text('$_anomalyClips',
            style: TextStyle(
              color: _anomalyClips > 0 ? _red : _green,
              fontSize: 10, fontWeight: FontWeight.w600)),
        ]),
      ]),
    );
  }

  Widget _driveVideoListWidget() {
    return Container(
      constraints: const BoxConstraints(maxHeight: 168),
      decoration: BoxDecoration(
        color: _bg1, borderRadius: BorderRadius.circular(7),
        border: Border.all(color: const Color(0xFF1565C0).withValues(alpha: 0.3)),
      ),
      child: ListView.builder(
        shrinkWrap: true,
        padding: const EdgeInsets.symmetric(vertical: 4),
        itemCount: _driveVideos.length,
        itemBuilder: (_, i) {
          final v        = _driveVideos[i];
          final fname    = v['filename'] as String;
          final fileId   = v['file_id'] as String;
          final sizeMb   = (v['size_mb'] as num).toDouble();
          final status   = _driveVideoStatus[fname] ?? 'waiting';
          final isActive = _videoName == fname;

          final IconData statusIcon;
          final Color    statusColor;
          switch (status) {
            case 'analyzing': statusIcon = Icons.hourglass_top; statusColor = _blue;
            case 'complete':  statusIcon = Icons.check_circle;  statusColor = _green;
            case 'error':     statusIcon = Icons.error;         statusColor = _red;
            default:          statusIcon = Icons.radio_button_unchecked; statusColor = _t2;
          }

          return GestureDetector(
            onTap: status == 'analyzing'
                ? null
                : () => _switchDriveVideo(fileId, fname),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 6),
              decoration: BoxDecoration(
                color: isActive ? _blue.withValues(alpha: 0.08) : Colors.transparent,
                border: isActive
                    ? Border(left: BorderSide(color: _blue, width: 2))
                    : null,
              ),
              child: Row(children: [
                Icon(statusIcon, size: 11, color: statusColor),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(fname,
                    style: TextStyle(
                      color: isActive ? _t0 : _t1,
                      fontSize: 10, fontWeight: isActive ? FontWeight.w600 : FontWeight.normal),
                    overflow: TextOverflow.ellipsis),
                ),
                Text('${sizeMb.toStringAsFixed(0)}M',
                  style: const TextStyle(color: _t2, fontSize: 9)),
              ]),
            ),
          );
        },
      ),
    );
  }

  Widget _stopAnalysisBtn() {
    return SizedBox(
      width: double.infinity,
      child: OutlinedButton.icon(
        onPressed: _emergencyStop,
        icon: const Icon(Icons.stop_circle_outlined, size: 15),
        label: const Text('Stop Analysis',
          style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600)),
        style: OutlinedButton.styleFrom(
          foregroundColor: _red,
          side: BorderSide(color: _red.withValues(alpha: 0.6)),
          backgroundColor: _red.withValues(alpha: 0.06),
          padding: const EdgeInsets.symmetric(vertical: 10),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        ),
      ),
    );
  }

  Widget _sideBtn(IconData icon, String label, {
    VoidCallback? onTap, Color color = _blue, bool loading = false}) {
    return SizedBox(
      width: double.infinity, height: 42,
      child: ElevatedButton.icon(
        onPressed: onTap,
        style: ElevatedButton.styleFrom(
          backgroundColor: color, disabledBackgroundColor: _bg2,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
          elevation: 0,
        ),
        icon: loading
            ? const SizedBox(width: 13, height: 13,
                child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
            : Icon(icon, color: Colors.white, size: 16),
        label: Text(label, style: const TextStyle(color: Colors.white, fontSize: 12)),
      ),
    );
  }

  Widget _label(String text) => Text(text.toUpperCase(),
    style: const TextStyle(color: _t2, fontSize: 10, letterSpacing: 1.2));

  Widget _dateBtn(String label, {VoidCallback? onTap}) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 9),
        decoration: BoxDecoration(
          color: _bg1, borderRadius: BorderRadius.circular(7),
          border: Border.all(color: _bd),
        ),
        child: Row(children: [
          const Icon(Icons.date_range_outlined, size: 13, color: _blue),
          const SizedBox(width: 8),
          Text(label, style: const TextStyle(color: _t1, fontSize: 11)),
        ]),
      ),
    );
  }

  Widget _statsPanel() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(11),
      decoration: BoxDecoration(
        color: _bg1, borderRadius: BorderRadius.circular(8),
        border: Border.all(color: _bd),
      ),
      child: Column(children: [
        _sRow('Total Videos',   '$_totalVideos'),
        _sRow('Processed',      '$_processedVideos'),
        _sRow('Anomaly Clips',  '$_anomalyClips',
          c: _anomalyClips > 0 ? _yellow : null),
        _sRow('Anomaly Vids.',  '$_anomalyVideos'),
        _sRow('Avg. Anomaly',
          '${(_avgAnomaly * 100).toStringAsFixed(1)}%'),
        if (_dominantClass != null)
          _sRow('Dominant Class', _dominantClass!,
            c: _classColor(_dominantClass!)),
        if (_processSec > 0) _sRow('Duration', '${_processSec}s'),
      ]),
    );
  }

  Widget _sRow(String k, String v, {Color? c}) => Padding(
    padding: const EdgeInsets.symmetric(vertical: 3),
    child: Row(children: [
      Expanded(
        flex: 5,
        child: Text(k,
          style: const TextStyle(color: _t1, fontSize: 11),
          overflow: TextOverflow.ellipsis)),
      const SizedBox(width: 4),
      Expanded(
        flex: 4,
        child: Text(v,
          style: TextStyle(color: c ?? _t0, fontSize: 11, fontWeight: FontWeight.w600),
          overflow: TextOverflow.ellipsis,
          textAlign: TextAlign.end)),
    ]),
  );

  Widget _timingCard() {
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: _bg1, borderRadius: BorderRadius.circular(8),
        border: Border.all(color: _bd),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        const Text('Last Clip Timing',
          style: TextStyle(color: _t2, fontSize: 10, letterSpacing: 0.5)),
        const SizedBox(height: 6),
        _tRow('CR-AE',   '$_craeMs ms',    _blue),
        _tRow('CNN',     '$_cnnMs ms',     _violet),
        _tRow('GradCAM', '$_heatmapMs ms', _amber),
        const Divider(color: _bd, height: 10),
        _tRow('Total',   '$_latencyMs ms', _t0, bold: true),
      ]),
    );
  }

  Widget _tRow(String k, String v, Color c, {bool bold = false}) => Padding(
    padding: const EdgeInsets.symmetric(vertical: 2),
    child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
      Text(k, style: const TextStyle(color: _t1, fontSize: 10)),
      Text(v, style: TextStyle(
        color: c, fontSize: 10,
        fontWeight: bold ? FontWeight.bold : FontWeight.w600)),
    ]),
  );

  // ─── Sag panel ──────────────────────────────────────────────────────────────

  Widget _rightPanel() {
    return Column(children: [
      Expanded(flex: 4, child: _videoSection()),
      if (_isAnalyzing || _videoName != null || _cameraMode || _driveMode) _timelineRow(),
      Expanded(flex: 3, child: _pipelineArea()),
    ]);
  }

  // ─── Video / Kamera görüntü alanı ────────────────────────────────────────────

  Widget _videoSection() {
    return Container(
      color: Colors.black,
      child: Stack(fit: StackFit.expand, children: [
        _videoContent(),

        // Video modu overlay'leri
        if (_videoName != null && !_cameraMode) ...[
          Positioned(top: 0, left: 0, right: 0, child: _videoTopBar()),
          Positioned(bottom: 0, left: 0, right: 0, child: _videoBottomBar()),
        ],

        // Kamera modu overlay'leri
        if (_cameraMode) ...[
          Positioned(top: 0, left: 0, right: 0, child: _cameraTopBar()),
          Positioned(bottom: 0, left: 0, right: 0, child: _videoBottomBar()),
          // Per-clip anomali bildirimleri (sağ üst köşe)
          Positioned(top: 40, right: 10,
            child: _clipAlertStack()),
        ],

        if (_videoPaused && !_cameraMode)
          Center(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.white24),
              ),
              child: const Row(mainAxisSize: MainAxisSize.min, children: [
                Icon(Icons.pause_circle_outline, color: Colors.white70, size: 20),
                SizedBox(width: 8),
                Text('PAUSED', style: TextStyle(
                  color: Colors.white70, fontSize: 12, letterSpacing: 1.5)),
              ]),
            ),
          ),
      ]),
    );
  }

  Widget _videoContent() {
    // ── Kamera modu: IR frame göster
    if (_cameraMode) {
      if (_cameraFrameB64 != null) {
        return Image.memory(_decode(_cameraFrameB64!),
          fit: BoxFit.contain, gaplessPlayback: true);
      }
      return Center(child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const SizedBox(width: 32, height: 32,
            child: CircularProgressIndicator(
              color: Color(0xFF0D7377), strokeWidth: 2)),
          const SizedBox(height: 12),
          Text(_cameraConnected ? 'Waiting for IR frame...' : 'Connecting to camera...',
            style: const TextStyle(color: _t1, fontSize: 12)),
        ],
      ));
    }

    // ── Drive modu: video yokken bekleme ekranı
    if (_driveMode && _videoName == null) {
      return Center(child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          FadeTransition(
            opacity: _pulseAnim,
            child: const Icon(Icons.cloud_outlined, size: 48, color: _blue)),
          const SizedBox(height: 14),
          Text(
            _driveConnected ? 'Monitoring Drive Folder' : 'Connecting to Drive...',
            style: const TextStyle(color: _t0, fontSize: 15, fontWeight: FontWeight.w500)),
          const SizedBox(height: 6),
          if (_driveStatusMsg != null)
            Text(_driveStatusMsg!,
              style: const TextStyle(color: _t1, fontSize: 11)),
        ],
      ));
    }

    // ── Video modu
    if (_videoName == null && !_isAnalyzing) {
      return Center(child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.thermostat, size: 52, color: _blue.withValues(alpha: 0.18)),
          const SizedBox(height: 14),
          const Text('Awaiting Analysis',
            style: TextStyle(color: _t2, fontSize: 16, fontWeight: FontWeight.w500)),
          const SizedBox(height: 6),
          const Text('Select dates and video from the left panel',
            style: TextStyle(color: _t2, fontSize: 12)),
        ],
      ));
    }
    if (_videoReady && _videoCtrl != null) {
      return Center(child: AspectRatio(
        aspectRatio: _videoCtrl!.value.aspectRatio,
        child: VideoPlayer(_videoCtrl!),
      ));
    }
    if (_heatmapB64 != null) {
      return Image.memory(_decode(_heatmapB64!),
        fit: BoxFit.contain, gaplessPlayback: true);
    }
    return Center(child: Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        const SizedBox(width: 30, height: 30,
          child: CircularProgressIndicator(color: _blue, strokeWidth: 2)),
        const SizedBox(height: 12),
        Text(_isAnalyzing ? 'Preparing video...' : 'Waiting',
          style: const TextStyle(color: _t1, fontSize: 12)),
      ],
    ));
  }

  Widget _cameraTopBar() {
    const camColor = Color(0xFF0D9488);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter, end: Alignment.bottomCenter,
          colors: [Color(0xCC000000), Colors.transparent],
        ),
      ),
      child: Row(children: [
        FadeTransition(opacity: _pulseAnim,
          child: const Icon(Icons.circle, size: 8, color: camColor)),
        const SizedBox(width: 6),
        const Text('IR LIVE', style: TextStyle(
          color: camColor, fontSize: 10, letterSpacing: 2,
          fontWeight: FontWeight.w700)),
        const SizedBox(width: 12),
        const Text('Infrared Camera',
          style: TextStyle(color: Colors.white54, fontSize: 11)),
        const Spacer(),
        if (_camClipIdx > 0)
          _tag('Clip $_camClipIdx'),
        const SizedBox(width: 6),
        if (_camAnomalyClips > 0)
          _tag('$_camAnomalyClips Anomaly',
            bg: _red.withValues(alpha: 0.3), fg: _red),
      ]),
    );
  }

  // Per-clip anomali bildirimleri (üst üste kartlar)
  Widget _clipAlertStack() {
    if (_clipAlerts.isEmpty) return const SizedBox.shrink();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.end,
      children: _clipAlerts.map((a) => Padding(
        padding: const EdgeInsets.only(bottom: 6),
        child: _clipAlertCard(a),
      )).toList(),
    );
  }

  Widget _clipAlertCard(_ClipAlert a) {
    final col = _classColor(a.classification);
    return Container(
      width: 230,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
      decoration: BoxDecoration(
        color: const Color(0xEE0D1117),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: col.withValues(alpha: 0.6)),
        boxShadow: [BoxShadow(
          color: col.withValues(alpha: 0.2), blurRadius: 12)],
      ),
      child: Row(children: [
        Icon(Icons.warning_amber_rounded, size: 16, color: col),
        const SizedBox(width: 8),
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Text('ANOMALY  Clip #${a.clipIdx}',
              style: const TextStyle(color: Colors.white54, fontSize: 9, letterSpacing: 0.8)),
            const Spacer(),
            Text('MSE ${a.mse.toStringAsFixed(4)}',
              style: const TextStyle(color: _yellow, fontSize: 9)),
          ]),
          const SizedBox(height: 3),
          Text(a.classification,
            style: TextStyle(color: col, fontSize: 11, fontWeight: FontWeight.w700)),
          Text('Score: ${(a.anomalyScore * 100).toStringAsFixed(1)}%',
            style: const TextStyle(color: Colors.white60, fontSize: 10)),
        ])),
      ]),
    );
  }

  Widget _videoTopBar() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter, end: Alignment.bottomCenter,
          colors: [Color(0xCC000000), Colors.transparent],
        ),
      ),
      child: Row(children: [
        if (_isAnalyzing) ...[
          FadeTransition(opacity: _pulseAnim,
            child: const Icon(Icons.circle, size: 8, color: _red)),
          const SizedBox(width: 6),
          const Text('LIVE',
            style: TextStyle(color: _red, fontSize: 10, letterSpacing: 2)),
          const SizedBox(width: 12),
        ],
        Expanded(
          child: Text(_videoName ?? '',
            style: const TextStyle(color: Colors.white54, fontSize: 11),
            overflow: TextOverflow.ellipsis),
        ),
        if (_totalClips > 0) _tag('Clip $_clipIdx / $_totalClips'),
      ]),
    );
  }

  Widget _videoBottomBar() {
    final mseCol = _exceeded ? _yellow : _green;
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 6, 12, 10),
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.bottomCenter, end: Alignment.topCenter,
          colors: [Color(0xCC000000), Colors.transparent],
        ),
      ),
      child: Row(children: [
        GestureDetector(
          onTap: _videoReady ? _toggleVideo : null,
          child: Container(
            padding: const EdgeInsets.all(6),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.15),
              shape: BoxShape.circle,
            ),
            child: Icon(_videoPaused ? Icons.play_arrow : Icons.pause,
              color: Colors.white, size: 17),
          ),
        ),
        const SizedBox(width: 10),
        _tag('Frame #$_frameNum'),
        const SizedBox(width: 6),
        _tag('MSE ${_rawMse.toStringAsFixed(4)}',
          bg: mseCol.withValues(alpha: 0.22), fg: mseCol),
        if (_exceeded && _classification != 'N/A') ...[
          const SizedBox(width: 6),
          _tag(_classification,
            bg: _classColor(_classification).withValues(alpha: 0.22),
            fg: _classColor(_classification)),
        ],
        const Spacer(),
        if (_latencyMs > 0)
          _tag('$_latencyMs ms', bg: Colors.white.withValues(alpha: 0.1)),
      ]),
    );
  }

  Widget _tag(String text, {Color? bg, Color? fg}) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
    decoration: BoxDecoration(
      color: bg ?? const Color(0x88000000),
      borderRadius: BorderRadius.circular(4),
    ),
    child: Text(text, style: TextStyle(color: fg ?? Colors.white70, fontSize: 10)),
  );

  // ─── Klip Timeline ───────────────────────────────────────────────────────────

  Widget _timelineRow() {
    return Container(
      height: 34,
      decoration: const BoxDecoration(
        color: _bg1,
        border: Border(top: BorderSide(color: _bd), bottom: BorderSide(color: _bd)),
      ),
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
        child: Row(children: [
          const Text('Clips  ',
            style: TextStyle(color: _t2, fontSize: 10)),
          ..._clipTimeline.asMap().entries.map((e) {
            return Tooltip(
              message: 'Clip ${e.key + 1}: ${e.value ? "Anomaly" : "Normal"}',
              child: Container(
                width: 10, height: 10,
                margin: const EdgeInsets.only(right: 2),
                decoration: BoxDecoration(
                  color: e.value ? _red : _green,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            );
          }),
          if (_isAnalyzing && _clipTimeline.isEmpty)
            const Text('Waiting...', style: TextStyle(color: _t2, fontSize: 10)),
        ]),
      ),
    );
  }

  // ─── Pipeline Alan ───────────────────────────────────────────────────────────

  Widget _pipelineArea() {
    return Container(
      color: _bg0,
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(10),
        child: Column(children: [
          _buildCraeSection(),
          _buildCnnSection(),
          _buildGradcamSection(),
        ]),
      ),
    );
  }

  // ─── CR-AE Bolumu ────────────────────────────────────────────────────────────

  Widget _buildCraeSection() {
    final barVal = (_rawMse / (_threshold * 2)).clamp(0.0, 1.0);
    final craeChild = Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: Row(children: [
          _stepCard('Frame\nExtract', '16 frame', Icons.layers_outlined, _blue, true),
          _arr(_blue),
          _stepCard('Resize', '128x128', Icons.aspect_ratio, _blue, true),
          _arr(_blue),
          _stepCard('Preprocess', 'Gray [0,1]', Icons.tune, _blue, true),
          _arr(_blue),
          _stepCard('CR-AE\nRecon.', 'LSTM-AE', Icons.memory, _blue, true),
          _arr(_blue),
          _mseCard(barVal),
          _arr(_blue),
          _decisionCard(),
        ]),
      ),
      const SizedBox(height: 10),
      Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Expanded(child: _mseChart()),
        if (_origFrameB64 != null || _reconFrameB64 != null) ...[
          const SizedBox(width: 12),
          _frameCompare(),
        ],
      ]),
    ]);
    return _section(
      title: 'CR-AE AUTOENCODER',
      color: _blue,
      timing: _craeMs > 0 ? '$_craeMs ms' : null,
      expanded: _craeOpen,
      onToggle: () => setState(() => _craeOpen = !_craeOpen),
      child: craeChild,
      expandedContent: craeChild,
    );
  }

  Widget _mseCard(double barVal) {
    return Container(
      width: 128,
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: _bg1, borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: _exceeded
              ? _yellow.withValues(alpha: 0.6)
              : _blue.withValues(alpha: 0.3)),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Icon(Icons.show_chart, size: 12, color: _exceeded ? _yellow : _blue),
          const SizedBox(width: 4),
          Text('MSE', style: TextStyle(color: _exceeded ? _yellow : _blue, fontSize: 10)),
        ]),
        const SizedBox(height: 4),
        Text(_rawMse.toStringAsFixed(6),
          style: TextStyle(
            color: _exceeded ? _yellow : _t0,
            fontSize: 14, fontWeight: FontWeight.bold, fontFamily: 'monospace')),
        const SizedBox(height: 5),
        ClipRRect(
          borderRadius: BorderRadius.circular(3),
          child: LinearProgressIndicator(
            value: barVal, minHeight: 4,
            backgroundColor: Colors.black12,
            valueColor: AlwaysStoppedAnimation(_exceeded ? _yellow : _blue),
          ),
        ),
        const SizedBox(height: 3),
        Text('Threshold: ${_threshold.toStringAsFixed(3)}',
          style: const TextStyle(color: _t2, fontSize: 9)),
      ]),
    );
  }

  Widget _decisionCard() {
    return Container(
      width: 100,
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: _exceeded ? _yellow.withValues(alpha: 0.07) : _green.withValues(alpha: 0.07),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: _exceeded ? _yellow.withValues(alpha: 0.5) : _green.withValues(alpha: 0.4)),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Icon(_exceeded ? Icons.warning_amber_rounded : Icons.check_circle_outline,
          size: 18, color: _exceeded ? _yellow : _green),
        const SizedBox(height: 4),
        Text(_exceeded ? 'THRESHOLD\nEXCEEDED' : 'NORMAL',
          style: TextStyle(
            color: _exceeded ? _yellow : _green,
            fontSize: 12, fontWeight: FontWeight.bold)),
        const SizedBox(height: 2),
        Text(_exceeded ? '-> CNN active' : '-> CNN skip',
          style: const TextStyle(color: _t2, fontSize: 9)),
      ]),
    );
  }

  Widget _mseChart() {
    final painter = _MseChartPainter(
      values: List.from(_mseHistory),
      threshold: _threshold,
      lineColor: _blue,
    );
    final hasData = _mseHistory.length >= 2;

    final smallChart = Container(
      height: 72,
      decoration: BoxDecoration(
        color: _bg1, borderRadius: BorderRadius.circular(6),
        border: Border.all(color: _bd),
      ),
      child: hasData
          ? Padding(
              padding: const EdgeInsets.all(6),
              child: CustomPaint(painter: painter))
          : const Center(
              child: Text('Awaiting data...',
                style: TextStyle(color: _t2, fontSize: 10))),
    );

    final largeChart = Column(children: [
      Padding(
        padding: const EdgeInsets.only(bottom: 8),
        child: Row(children: [
          const Text('MSE Time Series',
            style: TextStyle(color: _t1, fontSize: 13, fontWeight: FontWeight.w500)),
          const Spacer(),
          Text('Threshold: ${_threshold.toStringAsFixed(3)}',
            style: const TextStyle(color: _yellow, fontSize: 11)),
        ]),
      ),
      SizedBox(
        height: 240,
        child: hasData
            ? CustomPaint(painter: painter, size: Size.infinite)
            : const Center(child: Text('No data', style: TextStyle(color: _t2))),
      ),
    ]);

    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Row(children: [
        const Text('MSE Chart (live)',
          style: TextStyle(color: _t2, fontSize: 10)),
        const Spacer(),
        Text('threshold: ${_threshold.toStringAsFixed(3)}',
          style: const TextStyle(color: _t2, fontSize: 9)),
      ]),
      const SizedBox(height: 4),
      _tapToExpand(smallChart, expanded: largeChart, bgColor: _bg1),
    ]);
  }

  Widget _frameCompare() {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      const Text('Frame Comparison',
        style: TextStyle(color: _t2, fontSize: 10)),
      const SizedBox(height: 4),
      Row(children: [
        if (_origFrameB64 != null)
          Column(children: [
            _frameThumb(_origFrameB64!, _bd),
            const SizedBox(height: 3),
            const Text('Original', style: TextStyle(color: _t2, fontSize: 9)),
          ]),
        if (_origFrameB64 != null && _reconFrameB64 != null)
          const SizedBox(width: 6),
        if (_reconFrameB64 != null)
          Column(children: [
            _frameThumb(_reconFrameB64!, _blue.withValues(alpha: 0.45)),
            const SizedBox(height: 3),
            const Text('Reconstruct',
              style: TextStyle(color: _blue, fontSize: 9)),
          ]),
      ]),
    ]);
  }

  Widget _frameThumb(String b64, Color borderColor) {
    final thumb = Container(
      width: 82, height: 72,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(5),
        border: Border.all(color: borderColor),
      ),
      clipBehavior: Clip.antiAlias,
      child: Image.memory(_decode(b64), fit: BoxFit.cover, gaplessPlayback: true),
    );
    final fullImg = Image.memory(_decode(b64),
      fit: BoxFit.contain, gaplessPlayback: true);
    return _tapToExpand(thumb, expanded: fullImg);
  }

  // ─── CNN Bolumu ──────────────────────────────────────────────────────────────

  Widget _buildCnnSection() {
    final col = _exceeded ? _violet : _t2;
    final cnnChild = SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        _stepCard('Resize',    '112x112',               Icons.aspect_ratio, col, _exceeded),
        _arr(col),
        _stepCard('Preprocess', 'RGB [0,1]',            Icons.tune, col, _exceeded),
        _arr(col),
        _stepCard('Inference', _exceeded ? 'ResNet3D' : '---', Icons.psychology, col, _exceeded),
        _arr(col),
        _classDistCard(),
        if (_exceeded && _top2.isNotEmpty) ...[
          _arr(_violet),
          _top2Card(),
        ],
      ]),
    );
    return _section(
      title: '3D-CNN CLASSIFIER',
      color: col,
      timing: _exceeded && _cnnMs > 0 ? '$_cnnMs ms' : null,
      statusLabel: _exceeded ? 'ACTIVE' : 'SKIPPED',
      expanded: _cnnOpen,
      onToggle: () => setState(() => _cnnOpen = !_cnnOpen),
      child: cnnChild,
      expandedContent: cnnChild,
    );
  }

  Widget _classDistCard() {
    return Container(
      width: 185,
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: _bg1, borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: _exceeded ? _violet.withValues(alpha: 0.35) : _bd),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Icon(Icons.equalizer, size: 12, color: _exceeded ? _violet : _t2),
          const SizedBox(width: 4),
          Text('Class Distribution',
            style: TextStyle(color: _exceeded ? _violet : _t2, fontSize: 10)),
        ]),
        const SizedBox(height: 8),
        if (_exceeded && _classProbs.isNotEmpty)
          ..._classProbs.entries.map((e) => _classBar(e.key, e.value))
        else
          Text(
            _exceeded ? 'Awaiting class data...'
                      : 'CNN skipped: MSE threshold not exceeded.',
            style: const TextStyle(color: _t2, fontSize: 10)),
      ]),
    );
  }

  Widget _classBar(String name, double prob) {
    final isTop = _top2.isNotEmpty && _top2[0][0] == name;
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          SizedBox(width: 85,
            child: Text(name,
              style: TextStyle(color: isTop ? _t0 : _t1, fontSize: 10,
                fontWeight: isTop ? FontWeight.w600 : FontWeight.normal),
              overflow: TextOverflow.ellipsis)),
          const Spacer(),
          Text('${(prob * 100).toStringAsFixed(0)}%',
            style: TextStyle(
              color: isTop ? _classColor(name) : _t2,
              fontSize: 10, fontWeight: FontWeight.w600)),
        ]),
        const SizedBox(height: 3),
        ClipRRect(
          borderRadius: BorderRadius.circular(2),
          child: LinearProgressIndicator(
            value: prob, minHeight: 5,
            backgroundColor: Colors.black12,
            valueColor: AlwaysStoppedAnimation(
              isTop ? _classColor(name) : _t2),
          ),
        ),
      ]),
    );
  }

  Widget _top2Card() {
    return Container(
      width: 128,
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: _violet.withValues(alpha: 0.07),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: _violet.withValues(alpha: 0.35)),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        const Text('Top-2 Classes',
          style: TextStyle(color: _violet, fontSize: 10)),
        const SizedBox(height: 8),
        ..._top2.asMap().entries.map((e) {
          final i = e.key;
          final cls  = e.value;
          final name = cls[0].toString();
          final prob = (cls[1] as num).toDouble();
          return Padding(
            padding: const EdgeInsets.only(bottom: 6),
            child: Row(children: [
              Container(
                width: 16, height: 16,
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  color: i == 0 ? _classColor(name) : Colors.black12,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text('${i + 1}',
                  style: const TextStyle(color: Colors.white,
                    fontSize: 9, fontWeight: FontWeight.bold)),
              ),
              const SizedBox(width: 6),
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text(name,
                    style: TextStyle(color: i == 0 ? _t0 : _t1, fontSize: 10,
                      fontWeight: i == 0 ? FontWeight.w600 : FontWeight.normal),
                    overflow: TextOverflow.ellipsis),
                  Text('${(prob * 100).toStringAsFixed(1)}%',
                    style: TextStyle(
                      color: i == 0 ? _classColor(name) : _t2, fontSize: 9)),
                ]),
              ),
            ]),
          );
        }),
      ]),
    );
  }

  // ─── GradCAM Bolumu ──────────────────────────────────────────────────────────

  Widget _gradcamHeatmapBox(double w, double h) {
    final hasHm = _heatmapB64 != null && _heatmapB64!.isNotEmpty;
    final box = Container(
      width: w, height: h,
      decoration: BoxDecoration(
        color: _bg2, borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: hasHm ? _amber.withValues(alpha: 0.55) : _bd),
      ),
      clipBehavior: Clip.antiAlias,
      child: hasHm
          ? Image.memory(_decode(_heatmapB64!), fit: BoxFit.cover, gaplessPlayback: true)
          : const Center(child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.thermostat_outlined, color: _t2, size: 28),
                SizedBox(height: 6),
                Text('Awaiting Heatmap',
                  style: TextStyle(color: _t2, fontSize: 10),
                  textAlign: TextAlign.center),
              ],
            )),
    );
    if (!hasHm) return box;
    return _tapToExpand(
      box,
      expanded: Image.memory(_decode(_heatmapB64!),
        fit: BoxFit.contain, gaplessPlayback: true),
    );
  }

  Widget _gradcamColorLegend() {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      const Text('Color Scale (CR-AE Error Intensity)',
        style: TextStyle(color: _t2, fontSize: 9, letterSpacing: 0.3)),
      const SizedBox(height: 5),
      ClipRRect(
        borderRadius: BorderRadius.circular(3),
        child: Container(
          height: 10,
          decoration: const BoxDecoration(
            gradient: LinearGradient(colors: [
              Color(0xFF000000),
              Color(0xFF8B0000),
              Color(0xFFFF2200),
              Color(0xFFFFAA00),
              Color(0xFFFFFF00),
            ]),
          ),
        ),
      ),
      const SizedBox(height: 3),
      const Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text('Low', style: TextStyle(color: _t2, fontSize: 8)),
          Text('Mid', style: TextStyle(color: _t2, fontSize: 8)),
          Text('High', style: TextStyle(color: _t2, fontSize: 8)),
        ],
      ),
    ]);
  }

  Widget _buildGradcamSection() {
    final hasHm = _heatmapB64 != null && _heatmapB64!.isNotEmpty;

    final compactChild = Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
      _gradcamHeatmapBox(200, 120),
      const SizedBox(width: 14),
      Expanded(
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('CR-AE Spatial Error Map',
            style: TextStyle(color: _t1, fontSize: 11, fontWeight: FontWeight.w500)),
          const SizedBox(height: 4),
          const Text(
            'Regions with high reconstruction error\nare shown in warm tones.',
            style: TextStyle(color: _t2, fontSize: 10)),
          const SizedBox(height: 8),
          _gradcamColorLegend(),
          const SizedBox(height: 8),
          if (hasHm && _exceeded) ...[
            _infoRow('Detection', _classification, _classColor(_classification)),
            if (_confidence > 0)
              _infoRow('Confidence', '${(_confidence * 100).toStringAsFixed(1)}%', _green),
            _infoRow('MSE', _rawMse.toStringAsFixed(6), _yellow),
            _infoRow('Anomaly', '${(_anomalyScore * 100).toStringAsFixed(1)}%', _red),
          ] else
            const Text(
              'Heatmap appears when threshold is exceeded.',
              style: TextStyle(color: _t2, fontSize: 10)),
        ]),
      ),
    ]);

    final expandedChild = Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      _gradcamHeatmapBox(double.infinity, 280),
      const SizedBox(height: 14),
      _gradcamColorLegend(),
      const SizedBox(height: 14),
      const Text('Anomaly Region Analysis',
        style: TextStyle(color: _t1, fontSize: 12, fontWeight: FontWeight.w600)),
      const SizedBox(height: 8),
      const Text(
        'Warm-colored regions (yellow/red) indicate areas where the CR-AE model\n'
        'produced the highest reconstruction error. These represent the areas\n'
        'contributing most to the anomaly. Dark regions indicate areas perceived\n'
        'as normal by the model.',
        style: TextStyle(color: _t2, fontSize: 11, height: 1.6)),
      const SizedBox(height: 12),
      if (hasHm && _exceeded) ...[
        _infoRow('Detected Class', _classification, _classColor(_classification)),
        if (_confidence > 0)
          _infoRow('Model Confidence', '${(_confidence * 100).toStringAsFixed(1)}%', _green),
        _infoRow('Raw MSE Value', _rawMse.toStringAsFixed(7), _yellow),
        _infoRow('Anomaly Score', '${(_anomalyScore * 100).toStringAsFixed(1)}%', _red),
        _infoRow('Threshold', _threshold.toStringAsFixed(3), _t1),
        _infoRow('Exceeded', _exceeded ? 'YES' : 'NO',
          _exceeded ? _red : _green),
      ] else
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: _bg2, borderRadius: BorderRadius.circular(8),
            border: Border.all(color: _bd),
          ),
          child: const Text(
            'No anomaly region detected: MSE threshold not exceeded.\n'
            'Heatmap shows low activation.',
            style: TextStyle(color: _t2, fontSize: 11, height: 1.5)),
        ),
    ]);

    return _section(
      title: 'GRAD-CAM XAI',
      color: _amber,
      timing: _heatmapMs > 0 ? '$_heatmapMs ms' : null,
      expanded: _gradcamOpen,
      onToggle: () => setState(() => _gradcamOpen = !_gradcamOpen),
      child: compactChild,
      expandedContent: expandedChild,
    );
  }

  Widget _infoRow(String k, String v, Color c) => Padding(
    padding: const EdgeInsets.only(bottom: 5),
    child: Row(children: [
      Text('$k: ', style: const TextStyle(color: _t1, fontSize: 11)),
      Text(v, style: TextStyle(color: c, fontSize: 11, fontWeight: FontWeight.w600)),
    ]),
  );

  // ─── Section widget ──────────────────────────────────────────────────────────

  void _openSectionDialog(String title, Color color, Widget content) {
    showDialog(
      context: context,
      builder: (_) => Dialog(
        backgroundColor: _bg0,
        insetPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        child: SizedBox(
          width: double.infinity,
          height: MediaQuery.of(context).size.height * 0.82,
          child: Column(children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 13),
              decoration: BoxDecoration(
                color: _bg1,
                border: Border(bottom: BorderSide(color: color.withValues(alpha: 0.2))),
              ),
              child: Row(children: [
                Container(width: 8, height: 8,
                  decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
                const SizedBox(width: 10),
                Text(title, style: TextStyle(
                  color: color, fontSize: 13,
                  fontWeight: FontWeight.w600, letterSpacing: 0.5)),
                const Spacer(),
                GestureDetector(
                  onTap: () => Navigator.of(context, rootNavigator: true).pop(),
                  child: const Icon(Icons.close, size: 18, color: _t1)),
              ]),
            ),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(18),
                child: content,
              ),
            ),
          ]),
        ),
      ),
    );
  }

  Widget _section({
    required String title,
    required Color color,
    required bool expanded,
    required VoidCallback onToggle,
    required Widget child,
    required Widget expandedContent,
    String? timing,
    String? statusLabel,
  }) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      decoration: BoxDecoration(
        color: _bg1, borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withValues(alpha: 0.25)),
      ),
      child: Column(children: [
        InkWell(
          onTap: onToggle,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(10)),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            child: Row(children: [
              Container(width: 8, height: 8,
                decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
              const SizedBox(width: 8),
              Text(title, style: TextStyle(
                color: color, fontSize: 11,
                fontWeight: FontWeight.w600, letterSpacing: 0.7)),
              if (statusLabel != null) ...[
                const SizedBox(width: 8),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: color.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Text(statusLabel,
                    style: TextStyle(color: color, fontSize: 9)),
                ),
              ],
              const Spacer(),
              if (timing != null) ...[
                const Icon(Icons.timer_outlined, size: 11, color: _t2),
                const SizedBox(width: 3),
                Text(timing, style: const TextStyle(color: _t2, fontSize: 10)),
                const SizedBox(width: 6),
              ],
              GestureDetector(
                onTap: () => _openSectionDialog(title, color, expandedContent),
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                  child: Icon(Icons.open_in_full_rounded, size: 13, color: _t2),
                ),
              ),
              const SizedBox(width: 4),
              Icon(
                expanded
                    ? Icons.keyboard_arrow_up_rounded
                    : Icons.keyboard_arrow_down_rounded,
                size: 16, color: _t2),
            ]),
          ),
        ),
        if (expanded)
          Padding(
            padding: const EdgeInsets.fromLTRB(14, 2, 14, 14),
            child: child,
          ),
      ]),
    );
  }

  // ─── Helper widgets ──────────────────────────────────────────────────────────

  Widget _stepCard(String label, String val, IconData icon, Color col, bool active) {
    return Container(
      width: 84,
      padding: const EdgeInsets.all(9),
      decoration: BoxDecoration(
        color: active ? col.withValues(alpha: 0.08) : _bg2,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: active ? col.withValues(alpha: 0.3) : _bd),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Icon(icon, size: 13, color: active ? col : _t2),
        const SizedBox(height: 5),
        Text(label, style: TextStyle(
          color: active ? col : _t2, fontSize: 9, height: 1.3)),
        const SizedBox(height: 3),
        Text(val, style: TextStyle(
          color: active ? _t0 : _t2,
          fontSize: 10, fontWeight: FontWeight.w600)),
      ]),
    );
  }

  Widget _arr(Color col) => Padding(
    padding: const EdgeInsets.symmetric(horizontal: 3),
    child: Icon(Icons.chevron_right_rounded, size: 14,
      color: col.withValues(alpha: 0.4)),
  );

  Color _classColor(String cls) => _anomalyClassColor(cls);
}

// ─── Sınıf renk yardımcısı (top-level — tüm widget'lardan erişilebilir) ──────

Color _anomalyClassColor(String cls) {
  switch (cls.toLowerCase()) {
    case 'loitering':          return _yellow;
    case 'trespass':
    case 'trespassing':        return _red;
    case 'obj. aband.':
    case 'object_abandonment': return _violet;
    default:                   return _green;
  }
}

// ─── MSE Grafigi ─────────────────────────────────────────────────────────────

class _MseChartPainter extends CustomPainter {
  final List<double> values;
  final double threshold;
  final Color lineColor;

  const _MseChartPainter({
    required this.values,
    required this.threshold,
    required this.lineColor,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (values.length < 2) return;

    final maxVal = math.max(values.reduce(math.max) * 1.4, threshold * 1.8);
    final n = values.length;

    // Threshold yatay cizgisi (kesik)
    final tY = size.height * (1 - threshold / maxVal);
    final dashPaint = Paint()
      ..color = _yellow.withValues(alpha: 0.7)
      ..strokeWidth = 1;
    double dx = 0;
    while (dx < size.width) {
      canvas.drawLine(Offset(dx, tY), Offset(math.min(dx + 5, size.width), tY), dashPaint);
      dx += 10;
    }

    // Gradient dolgu
    final path = Path();
    for (int i = 0; i < n; i++) {
      final x = i / (n - 1) * size.width;
      final y = size.height * (1 - values[i] / maxVal);
      if (i == 0) { path.moveTo(x, y); } else { path.lineTo(x, y); }
    }
    final fillPath = Path.from(path)
      ..lineTo(size.width, size.height)
      ..lineTo(0, size.height)
      ..close();
    canvas.drawPath(fillPath, Paint()
      ..shader = LinearGradient(
        begin: Alignment.topCenter, end: Alignment.bottomCenter,
        colors: [lineColor.withValues(alpha: 0.28), lineColor.withValues(alpha: 0.0)],
      ).createShader(Rect.fromLTWH(0, 0, size.width, size.height)));

    // Ana cizgi
    canvas.drawPath(path, Paint()
      ..color = lineColor
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round);

    // Anomali noktalari (esik ustunde kirmizi dot)
    for (int i = 0; i < n; i++) {
      if (values[i] > threshold) {
        final x = i / (n - 1) * size.width;
        final y = size.height * (1 - values[i] / maxVal);
        canvas.drawCircle(Offset(x, y), 3, Paint()..color = _yellow);
      }
    }
  }

  @override
  bool shouldRepaint(_MseChartPainter _) => true;
}

// ─── Drive Yükleniyor Dialog ──────────────────────────────────────────────────

class _LoadingDialog extends StatelessWidget {
  final String message;
  const _LoadingDialog({required this.message});

  @override
  Widget build(BuildContext context) {
    return Dialog(
      backgroundColor: _bg1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 24),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          const SizedBox(width: 18, height: 18,
            child: CircularProgressIndicator(strokeWidth: 2, color: _blue)),
          const SizedBox(width: 16),
          Text(message, style: const TextStyle(color: _t1, fontSize: 13)),
        ]),
      ),
    );
  }
}

// ─── Drive Video Seç Dialog ───────────────────────────────────────────────────

class _DriveSelectDialog extends StatelessWidget {
  final String folderPath;
  final List<Map<String, dynamic>> videos;
  final VoidCallback onSelectAll;
  final void Function(String fileId, String filename) onSelectVideo;

  const _DriveSelectDialog({
    required this.folderPath,
    required this.videos,
    required this.onSelectAll,
    required this.onSelectVideo,
  });

  @override
  Widget build(BuildContext context) {
    return Dialog(
      backgroundColor: _bg1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: SizedBox(
        width: 440,
        child: Column(mainAxisSize: MainAxisSize.min, children: [

          // ── Header
          Container(
            padding: const EdgeInsets.fromLTRB(18, 16, 14, 14),
            decoration: const BoxDecoration(
              border: Border(bottom: BorderSide(color: _bd)),
            ),
            child: Row(children: [
              const Icon(Icons.cloud_outlined, color: _blue, size: 18),
              const SizedBox(width: 10),
              const Expanded(
                child: Text('Drive Folder Monitor',
                  style: TextStyle(color: _t0, fontSize: 14,
                    fontWeight: FontWeight.w600)),
              ),
              IconButton(
                onPressed: () => Navigator.of(context).pop(),
                icon: const Icon(Icons.close, size: 18, color: _t1),
                padding: EdgeInsets.zero, constraints: const BoxConstraints(),
              ),
            ]),
          ),

          // ── Folder path
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 10),
            decoration: const BoxDecoration(
              color: _bg2,
              border: Border(bottom: BorderSide(color: _bd)),
            ),
            child: Row(children: [
              const Icon(Icons.folder_outlined, size: 13, color: _blue),
              const SizedBox(width: 8),
              Expanded(
                child: Text(folderPath,
                  style: const TextStyle(color: _t1, fontSize: 11,
                    fontFamily: 'monospace'),
                  overflow: TextOverflow.ellipsis),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: _blue.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text('${videos.length} video',
                  style: const TextStyle(color: _blue, fontSize: 10)),
              ),
            ]),
          ),

          // ── Video list
          ConstrainedBox(
            constraints: const BoxConstraints(maxHeight: 280),
            child: ListView.builder(
              shrinkWrap: true,
              itemCount: videos.length,
              itemBuilder: (_, i) {
                final v        = videos[i];
                final fname    = v['filename'] as String;
                final fileId   = v['file_id'] as String;
                final sizeMb   = (v['size_mb'] as num).toDouble();
                final subfolder = (v['subfolder'] as String?) ?? '';
                return ListTile(
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 16, vertical: 2),
                  leading: Container(
                    width: 32, height: 32,
                    decoration: BoxDecoration(
                      color: _blue.withValues(alpha: 0.10),
                      borderRadius: BorderRadius.circular(6),
                    ),
                    child: const Icon(Icons.videocam_outlined,
                      color: _blue, size: 16),
                  ),
                  title: Text(fname,
                    style: const TextStyle(color: _t0, fontSize: 12)),
                  subtitle: Text(
                    subfolder.isNotEmpty
                        ? '$subfolder  •  ${sizeMb.toStringAsFixed(1)} MB'
                        : '${sizeMb.toStringAsFixed(1)} MB',
                    style: TextStyle(
                      color: subfolder.isNotEmpty ? _anomalyClassColor(subfolder) : _t2,
                      fontSize: 10)),
                  trailing: const Icon(Icons.play_arrow_rounded,
                    color: _t2, size: 18),
                  onTap: () => onSelectVideo(fileId, fname),
                );
              },
            ),
          ),

          // ── Analyze all button
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
            child: SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: onSelectAll,
                icon: const Icon(Icons.cloud_download_outlined, size: 16),
                label: const Text('Analyze All',
                  style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
                style: ElevatedButton.styleFrom(
                  backgroundColor: _blue,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 13),
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(9)),
                ),
              ),
            ),
          ),
        ]),
      ),
    );
  }
}

// ─── Video Listesi Dialog ─────────────────────────────────────────────────────

class _VideoListDialog extends StatelessWidget {
  final List<Map<String, dynamic>> videos;
  const _VideoListDialog({required this.videos});

  @override
  Widget build(BuildContext context) {
    return Dialog(
      backgroundColor: _bg1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: SizedBox(
        width: 400,
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
            child: Row(children: [
              const Icon(Icons.video_library_outlined, color: _blue, size: 18),
              const SizedBox(width: 8),
              const Text('Select Video',
                style: TextStyle(color: _t0, fontSize: 14,
                  fontWeight: FontWeight.w600)),
              const Spacer(),
              IconButton(
                onPressed: () => Navigator.of(context).pop(),
                icon: const Icon(Icons.close, size: 18, color: _t1),
              ),
            ]),
          ),
          const Divider(color: _bd, height: 1),
          ConstrainedBox(
            constraints: const BoxConstraints(maxHeight: 300),
            child: ListView.builder(
              shrinkWrap: true,
              itemCount: videos.length,
              itemBuilder: (_, i) {
                final v = videos[i];
                final name   = v['filename'] ?? v['name'] ?? 'unknown';
                final folder = v['date_folder'] ?? v['date'] ?? '';
                return ListTile(
                  leading: Container(
                    width: 32, height: 32,
                    decoration: BoxDecoration(
                      color: _blue.withValues(alpha: 0.12),
                      borderRadius: BorderRadius.circular(6),
                    ),
                    child: const Icon(Icons.videocam, color: _blue, size: 16),
                  ),
                  title: Text(name,
                    style: const TextStyle(color: _t0, fontSize: 13)),
                  subtitle: Text(folder,
                    style: const TextStyle(color: _t2, fontSize: 11)),
                  onTap: () => Navigator.of(context).pop(name),
                );
              },
            ),
          ),
        ]),
      ),
    );
  }
}
