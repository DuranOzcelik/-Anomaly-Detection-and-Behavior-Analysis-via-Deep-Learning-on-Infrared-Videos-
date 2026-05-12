import 'package:flutter/material.dart';

class VideoHeatmapOverlay extends StatefulWidget {
  final String heatmapBase64;
  final String filename;
  final double opacity;

  const VideoHeatmapOverlay({
    super.key,
    required this.heatmapBase64,
    required this.filename,
    this.opacity = 0.6,
  });

  @override
  State<VideoHeatmapOverlay> createState() => _VideoHeatmapOverlayState();
}

class _VideoHeatmapOverlayState extends State<VideoHeatmapOverlay> {
  late double _opacityValue;

  @override
  void initState() {
    super.initState();
    _opacityValue = widget.opacity;
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        // Heatmap görüntü
        Center(
          child: Image.memory(
            _base64ToBytes(widget.heatmapBase64),
            fit: BoxFit.contain,
            width: 640,
            height: 480,
          ),
        ),
        // Opacity kontrolü
        Positioned(
          bottom: 16,
          left: 16,
          right: 16,
          child: Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.7),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      'Heatmap Opaklığı',
                      style: Theme.of(context).textTheme.labelSmall?.copyWith(
                            color: Colors.white,
                          ),
                    ),
                    Text(
                      '${(_opacityValue * 100).toStringAsFixed(0)}%',
                      style: Theme.of(context).textTheme.labelSmall?.copyWith(
                            color: Colors.white,
                          ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Slider(
                  value: _opacityValue,
                  onChanged: (value) {
                    setState(() {
                      _opacityValue = value;
                    });
                  },
                  min: 0.0,
                  max: 1.0,
                  activeColor: Colors.blue,
                  inactiveColor: Colors.grey,
                ),
              ],
            ),
          ),
        ),
        // Dosya adı
        Positioned(
          top: 16,
          left: 16,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.7),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              widget.filename,
              style: Theme.of(context).textTheme.labelSmall?.copyWith(
                    color: Colors.white,
                  ),
              overflow: TextOverflow.ellipsis,
              maxLines: 1,
            ),
          ),
        ),
      ],
    );
  }

  List<int> _base64ToBytes(String base64String) {
    // Remove data URI prefix if present
    final String cleanBase64 = base64String.contains(',')
        ? base64String.split(',').last
        : base64String;

    return base64Decode(cleanBase64);
  }
}
