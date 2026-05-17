import 'dart:async';
import 'dart:convert';
import 'dart:developer' as developer;
import 'package:web_socket_channel/web_socket_channel.dart';

class ProcessingMessage {
  final String type;
  final String? filename;
  final String? dateFolder;
  final String? videoUrl;
  final int? frameNumber;
  final int? clipIndex;
  final int? totalClips;
  final double? anomalyScore;
  final String? classification;
  final double? confidence;
  final int? latencyMs;
  final String? heatmapBase64;
  final String? errorMessage;
  final int? totalVideos;

  final double? avgAnomalyScore;
  final String? dominantClass;
  final Map<String, int>? classDistribution;
  final int? totalFramesProcessed;
  final int? framesSampled;
  final double? avgAnomalyAcrossAll;
  final int? totalVideosWithAnomalies;
  final int? processingTimeSeconds;

  // Cascade pipeline
  final double? rawMse;
  final bool? thresholdExceeded;

  // Per-step timings
  final int? craeTimeMs;
  final int? cnnTimeMs;
  final int? heatmapTimeMs;

  // CNN class probabilities
  final Map<String, double>? allClassProbs;
  final List<List<dynamic>>? top2Classes;

  // CR-AE frame images
  final String? reconFrameBase64;
  final String? origFrameBase64;

  // Camera-specific
  final String? cameraFrameBase64;
  final String? cameraId;
  final int?    anomalyClips;

  // Drive-specific
  final String? message;

  ProcessingMessage({
    required this.type,
    this.filename,
    this.dateFolder,
    this.videoUrl,
    this.frameNumber,
    this.clipIndex,
    this.totalClips,
    this.anomalyScore,
    this.classification,
    this.confidence,
    this.latencyMs,
    this.heatmapBase64,
    this.errorMessage,
    this.totalVideos,
    this.avgAnomalyScore,
    this.dominantClass,
    this.classDistribution,
    this.totalFramesProcessed,
    this.framesSampled,
    this.avgAnomalyAcrossAll,
    this.totalVideosWithAnomalies,
    this.processingTimeSeconds,
    this.rawMse,
    this.thresholdExceeded,
    this.craeTimeMs,
    this.cnnTimeMs,
    this.heatmapTimeMs,
    this.allClassProbs,
    this.top2Classes,
    this.reconFrameBase64,
    this.origFrameBase64,
    this.cameraFrameBase64,
    this.cameraId,
    this.anomalyClips,
    this.message,
  });

  factory ProcessingMessage.fromJson(Map<String, dynamic> json) {
    Map<String, double>? allClassProbs;
    if (json['all_class_probs'] != null) {
      allClassProbs = Map<String, double>.from(
        (json['all_class_probs'] as Map).map(
          (k, v) => MapEntry(k as String, (v as num).toDouble()),
        ),
      );
    }

    List<List<dynamic>>? top2Classes;
    if (json['top2_classes'] != null) {
      top2Classes = (json['top2_classes'] as List)
          .map((e) => e as List<dynamic>)
          .toList();
    }

    return ProcessingMessage(
      type: json['type'] ?? '',
      filename: json['filename'],
      dateFolder: json['date_folder'],
      videoUrl: json['video_url'],
      frameNumber: json['frame_number'],
      clipIndex: json['clip_index'],
      totalClips: json['total_clips'],
      anomalyScore: (json['anomaly_score'] as num?)?.toDouble(),
      classification: json['classification'],
      confidence: (json['confidence'] as num?)?.toDouble(),
      latencyMs: json['latency_ms'],
      heatmapBase64: json['heatmap_base64'],
      errorMessage: json['error_message'],
      totalVideos: json['total_videos'],
      avgAnomalyScore: (json['avg_anomaly_score'] as num?)?.toDouble(),
      dominantClass: json['dominant_class'],
      classDistribution: json['class_distribution'] != null
          ? Map<String, int>.from(json['class_distribution'] as Map)
          : null,
      totalFramesProcessed: json['total_frames_processed'],
      framesSampled: json['frames_sampled'],
      avgAnomalyAcrossAll: (json['avg_anomaly_across_all'] as num?)?.toDouble(),
      totalVideosWithAnomalies: json['total_videos_with_anomalies'],
      processingTimeSeconds: json['processing_time_seconds'],
      rawMse: (json['raw_mse'] as num?)?.toDouble(),
      thresholdExceeded: json['threshold_exceeded'] as bool?,
      craeTimeMs: json['crae_time_ms'],
      cnnTimeMs: json['cnn_time_ms'],
      heatmapTimeMs: json['heatmap_time_ms'],
      allClassProbs: allClassProbs,
      top2Classes: top2Classes,
      reconFrameBase64: json['recon_frame_base64'],
      origFrameBase64: json['orig_frame_base64'],
      cameraFrameBase64: json['frame_base64'],
      cameraId: json['camera_id'],
      anomalyClips: json['anomaly_clips'],
      message: json['message'],
    );
  }
}

class WebSocketService {
  WebSocketChannel? _channel;
  bool _isConnected = false;
  StreamSubscription? _subscription;

  final String _baseUrl;
  final String _path;
  final StreamController<ProcessingMessage> _messageController =
      StreamController<ProcessingMessage>.broadcast();

  WebSocketService({
    String baseUrl = 'ws://127.0.0.1:8000',
    String path    = '/ws',
  })  : _baseUrl = baseUrl,
        _path    = path;

  bool get isConnected => _isConnected;
  Stream<ProcessingMessage> get messages => _messageController.stream;

  Future<void> connect() async {
    try {
      _channel = WebSocketChannel.connect(Uri.parse('$_baseUrl$_path'));
      await _channel!.ready.timeout(const Duration(seconds: 10));
      _isConnected = true;

      _subscription = _channel!.stream.listen(
        _handleMessage,
        onError: (error) {
          _isConnected = false;
          _messageController.add(
            ProcessingMessage(
              type: 'error',
              errorMessage: 'WebSocket error: $error',
            ),
          );
        },
        onDone: () {
          _isConnected = false;
          developer.log('WebSocket connection closed');
        },
      );
    } on TimeoutException {
      _isConnected = false;
      throw Exception('WebSocket connection timed out: backend unreachable ($_baseUrl)');
    } catch (e) {
      _isConnected = false;
      throw Exception('WebSocket connection error: $e');
    }
  }

  void _handleMessage(dynamic raw) {
    try {
      final data = jsonDecode(raw as String) as Map<String, dynamic>;
      _messageController.add(ProcessingMessage.fromJson(data));
    } catch (e) {
      _messageController.add(
        ProcessingMessage(type: 'error', errorMessage: 'Failed to parse message: $e'),
      );
    }
  }

  Future<void> startAnalysis({required String startDate, required String endDate}) async {
    if (!_isConnected || _channel == null) {
      throw Exception('WebSocket not connected');
    }
    _channel!.sink.add(jsonEncode({
      'action': 'start_analysis',
      'start_date': startDate,
      'end_date': endDate,
    }));
  }

  Future<void> startVideoProcessing({required String filename}) async {
    if (!_isConnected || _channel == null) {
      throw Exception('WebSocket not connected');
    }
    _channel!.sink.add(jsonEncode({'action': 'start_video', 'filename': filename}));
  }

  Future<void> stopAnalysis() async {
    if (!_isConnected || _channel == null) return;
    _channel!.sink.add(jsonEncode({'action': 'stop'}));
  }

  Future<void> startDriveWatch({String? fileId, String? filename}) async {
    if (!_isConnected || _channel == null) {
      throw Exception('Drive WebSocket not connected');
    }
    final msg = <String, dynamic>{'action': 'start_drive_watch'};
    if (fileId != null)   msg['file_id']  = fileId;
    if (filename != null) msg['filename'] = filename;
    _channel!.sink.add(jsonEncode(msg));
  }

  Future<void> stopDriveWatch() async {
    if (!_isConnected || _channel == null) return;
    _channel!.sink.add(jsonEncode({'action': 'stop_drive_watch'}));
  }

  Future<void> startCamera({int cameraIndex = 0}) async {
    if (!_isConnected || _channel == null) {
      throw Exception('Camera WebSocket not connected');
    }
    _channel!.sink.add(jsonEncode({
      'action': 'start_camera',
      'camera_index': cameraIndex,
    }));
  }

  Future<void> stopCamera() async {
    if (!_isConnected || _channel == null) return;
    _channel!.sink.add(jsonEncode({'action': 'stop_camera'}));
  }

  void dispose() {
    _subscription?.cancel();
    _channel?.sink.close();
    _messageController.close();
    _isConnected = false;
  }
}
