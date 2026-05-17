import 'dart:io';
import 'dart:async';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:developer' as developer;

class ProcessVideoResponse {
  final String jobId;
  final String filename;
  final String classification;
  final double confidence;
  final int latencyMs;
  final String heatmapBase64;

  ProcessVideoResponse({
    required this.jobId,
    required this.filename,
    required this.classification,
    required this.confidence,
    required this.latencyMs,
    required this.heatmapBase64,
  });

  factory ProcessVideoResponse.fromJson(Map<String, dynamic> json) {
    return ProcessVideoResponse(
      jobId: json['job_id'] ?? '',
      filename: json['filename'] ?? '',
      classification: json['classification'] ?? 'Unknown',
      confidence: (json['confidence'] ?? 0.0).toDouble(),
      latencyMs: json['latency_ms'] ?? 0,
      heatmapBase64: json['heatmap'] ?? '',
    );
  }
}

class ApiService {
  final String baseUrl;

  ApiService({this.baseUrl = 'http://127.0.0.1:8000'});

  Future<List<dynamic>> listVideos(String startDate, String endDate) async {
    try {
      final response = await http
          .get(
            Uri.parse('$baseUrl/videos?start_date=$startDate&end_date=$endDate'),
          )
          .timeout(const Duration(seconds: 15));

      if (response.statusCode == 200) {
        return jsonDecode(response.body) as List<dynamic>;
      }
      throw Exception('Failed to list videos: status ${response.statusCode}');
    } on SocketException catch (e) {
      developer.log('Connection error', error: e);
      throw Exception('Connection error: backend unreachable ($baseUrl)');
    } catch (e) {
      developer.log('listVideos error', error: e);
      throw Exception('listVideos error: $e');
    }
  }

  Future<ProcessVideoResponse> uploadVideo(File videoFile) async {
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/process-video'),
      );

      request.files.add(
        await http.MultipartFile.fromPath('file', videoFile.path),
      );

      final response = await request.send().timeout(const Duration(seconds: 30));
      final responseBody = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        return ProcessVideoResponse.fromJson(jsonDecode(responseBody));
      }
      throw Exception('Video upload failed: status ${response.statusCode}');
    } on SocketException catch (e) {
      developer.log('Connection error', error: e);
      throw Exception('Connection error: backend unreachable ($baseUrl)');
    } on TimeoutException catch (e) {
      developer.log('Upload timeout', error: e);
      throw Exception('Upload timed out: video may be too large');
    } catch (e) {
      developer.log('uploadVideo error', error: e);
      throw Exception('Upload error: $e');
    }
  }

  Future<ProcessVideoResponse> getResults(String jobId) async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/results/$jobId'))
          .timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        return ProcessVideoResponse.fromJson(jsonDecode(response.body));
      }
      throw Exception('Failed to get results: status ${response.statusCode}');
    } on SocketException catch (e) {
      developer.log('Connection error', error: e);
      throw Exception('Connection error: backend unreachable');
    } catch (e) {
      developer.log('getResults error', error: e);
      throw Exception('getResults error: $e');
    }
  }

  Future<Map<String, dynamic>> listDriveVideos() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/drive/videos'))
          .timeout(const Duration(seconds: 30));
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      }
      throw Exception('Failed to list Drive videos: ${response.statusCode}');
    } on SocketException {
      throw Exception('Connection error: backend unreachable ($baseUrl)');
    } catch (e) {
      rethrow;
    }
  }

  Future<bool> healthCheck() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (e) {
      developer.log('Health check failed', error: e);
      return false;
    }
  }
}
