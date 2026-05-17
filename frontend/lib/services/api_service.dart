import 'dart:io';
import 'dart:async'; // -> TimeoutException hatasını çözen kritik import
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

  /// Seçilen tarih aralığındaki videoları listeler
  Future<List<dynamic>> listVideos(String startDate, String endDate) async {
    try {
      print('[🔍 API] Videolar listeleniyor: $startDate - $endDate');
      final response = await http
          .get(
            Uri.parse(
              '$baseUrl/videos?start_date=$startDate&end_date=$endDate',
            ),
          )
          .timeout(const Duration(seconds: 15));

      if (response.statusCode == 200) {
        final List<dynamic> jsonResponse = jsonDecode(response.body);
        print('[✓ API] Videolar alındı: ${jsonResponse.length} adet');
        return jsonResponse;
      } else {
        final error =
            'Video listesi alma başarısız: Status ${response.statusCode}';
        print('[❌ API] $error');
        developer.log(error);
        throw Exception(error);
      }
    } on SocketException catch (e) {
      final error = '📡 Bağlantı hatası: Backend erişilemiyor ($baseUrl)';
      print('[❌ API] $error - $e');
      developer.log(error, error: e);
      throw Exception(error);
    } catch (e) {
      final error = 'Video listesi alma hatası: $e';
      print('[❌ API] $error');
      developer.log(error, error: e);
      throw Exception(error);
    }
  }

  /// Tekli video yükleme ve işleme fonksiyonu
  Future<ProcessVideoResponse> uploadVideo(File videoFile) async {
    try {
      print('[📤 API] Video yükleniyor: ${videoFile.path}');
      developer.log('Video yükleniyor: ${videoFile.path}');

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/process-video'),
      );

      request.files.add(
        await http.MultipartFile.fromPath('file', videoFile.path),
      );

      print('[🔄 API] İstek gönderiliyor...');
      final response = await request.send().timeout(
        const Duration(seconds: 30),
      );
      final responseBody = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        final jsonResponse = jsonDecode(responseBody);
        print('[✓ API] Video başarıyla işlendi: ${jsonResponse['job_id']}');
        developer.log('Video başarıyla işlendi');
        return ProcessVideoResponse.fromJson(jsonResponse);
      } else {
        final error = 'Video yükleme başarısız: Status ${response.statusCode}';
        print('[❌ API] $error');
        developer.log(error);
        throw Exception(error);
      }
    } on SocketException catch (e) {
      final error = '📡 Bağlantı hatası: Backend erişilemiyor ($baseUrl)';
      print('[❌ API] $error - $e');
      developer.log(error, error: e);
      throw Exception(error);
    } on TimeoutException catch (e) {
      final error = '⏱ İşlem zaman aşımı: Video çok büyük olabilir';
      print('[❌ API] $error - $e');
      developer.log(error, error: e);
      throw Exception(error);
    } catch (e) {
      final error = 'Video yükleme hatası: $e';
      print('[❌ API] $error');
      developer.log(error, error: e);
      throw Exception(error);
    }
  }

  /// Geçmiş bir işin (Job) sonuçlarını çeker
  Future<ProcessVideoResponse> getResults(String jobId) async {
    try {
      print('[🔍 API] Sonuçlar alınıyor: $jobId');
      final response = await http
          .get(Uri.parse('$baseUrl/results/$jobId'))
          .timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        final jsonResponse = jsonDecode(response.body);
        print('[✓ API] Sonuçlar alındı: $jobId');
        return ProcessVideoResponse.fromJson(jsonResponse);
      } else {
        final error = 'Sonuç alma başarısız: Status ${response.statusCode}';
        print('[❌ API] $error');
        developer.log(error);
        throw Exception(error);
      }
    } on SocketException catch (e) {
      final error = '📡 Bağlantı hatası: Backend erişilemiyor';
      print('[❌ API] $error - $e');
      developer.log(error, error: e);
      throw Exception(error);
    } catch (e) {
      final error = 'Sonuç alma hatası: $e';
      print('[❌ API] $error');
      developer.log(error, error: e);
      throw Exception(error);
    }
  }

  /// DRIVE_WATCH_FOLDER_PATH klasöründeki videoları listeler
  Future<Map<String, dynamic>> listDriveVideos() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/drive/videos'))
          .timeout(const Duration(seconds: 30));
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      }
      throw Exception('Drive video listesi alınamadı: ${response.statusCode}');
    } on SocketException {
      throw Exception('Bağlantı hatası: Backend erişilemiyor ($baseUrl)');
    } catch (e) {
      rethrow;
    }
  }

  /// Sunucu durumunu kontrol eder
  Future<bool> healthCheck() async {
    try {
      print('[🏥 API] Health check yapılıyor...');
      final response = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 5));

      if (response.statusCode == 200) {
        print('[✓ API] Backend sağlıklı');
        return true;
      } else {
        print('[⚠ API] Health check başarısız: ${response.statusCode}');
        return false;
      }
    } catch (e) {
      final error = '❌ Backend bağlantı hatası: $baseUrl';
      print('[❌ API] $error - $e');
      developer.log(error, error: e);
      return false;
    }
  }
}
