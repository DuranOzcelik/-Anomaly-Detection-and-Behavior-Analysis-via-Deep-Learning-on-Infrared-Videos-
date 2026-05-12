import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';

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

  ApiService({this.baseUrl = 'http://localhost:8000'});

  Future<ProcessVideoResponse> uploadVideo(File videoFile) async {
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/process-video'),
      );

      request.files.add(
        await http.MultipartFile.fromPath('file', videoFile.path),
      );

      final response = await request.send();
      final responseBody = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        final jsonResponse = jsonDecode(responseBody);
        return ProcessVideoResponse.fromJson(jsonResponse);
      } else {
        throw Exception('Failed to upload video: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error uploading video: $e');
    }
  }

  Future<ProcessVideoResponse> getResults(String jobId) async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/results/$jobId'),
      );

      if (response.statusCode == 200) {
        final jsonResponse = jsonDecode(response.body);
        return ProcessVideoResponse.fromJson(jsonResponse);
      } else {
        throw Exception('Failed to get results: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error getting results: $e');
    }
  }

  Future<bool> healthCheck() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/health'),
      ).timeout(const Duration(seconds: 5));

      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
}
