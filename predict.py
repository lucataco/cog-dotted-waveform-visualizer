from cog import BasePredictor, Input, Path
import cv2
import numpy as np
import subprocess


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.frame = None

    def draw_symmetric_dots(
        self, frame, x_coords, y1_coords, y2_coords, radius, color_full, color_half
    ):
        """Draw symmetric dot pairs directly on the frame"""
        h, w = frame.shape[:2]
        y1_coords = np.clip(y1_coords, radius, h - radius)
        y2_coords = np.clip(y2_coords, radius, h - radius)

        # Randomly assign full or half brightness
        mask = np.random.random(len(x_coords)) < 0.5

        # Draw full brightness dots
        for x, y1, y2 in zip(x_coords[mask], y1_coords[mask], y2_coords[mask]):
            cv2.circle(frame, (int(x), int(y1)), radius, color_full, -1)
            cv2.circle(frame, (int(x), int(y2)), radius, color_full, -1)

        # Draw half brightness dots
        half_mask = ~mask
        for x, y1, y2 in zip(
            x_coords[half_mask], y1_coords[half_mask], y2_coords[half_mask]
        ):
            cv2.circle(frame, (int(x), int(y1)), radius, color_half, -1)
            cv2.circle(frame, (int(x), int(y2)), radius, color_half, -1)

    def predict(
        self,
        audio_file: Path = Input(description="Input audio file"),
        dot_size: int = Input(description="Size of dots in pixels", default=6),
        dot_spacing: int = Input(
            description="Spacing between dots in pixels", default=6
        ),
        height: int = Input(
            description="Height of the output video in pixels",
            default=720,
            ge=100,
            le=1280,
        ),
        width: int = Input(
            description="Width of the output video in pixels",
            default=1280,
            ge=100,
            le=1280,
        ),
        max_height: int = Input(
            description="Maximum height of visualization as a percentage",
            default=30,
            ge=5,
            le=100,
        ),
        dot_color: str = Input(
            description="Dot color in hex format", default="#00FFFF"
        ),
        fps: int = Input(description="Frames per second", default=10, ge=1, le=30),
    ) -> Path:
        """Run a single prediction on the model"""

        print("Loading audio file...")
        # Decode audio to raw PCM via ffmpeg — lightweight, no librosa overhead
        sr = 22050  # Sufficient for amplitude visualization
        decode_cmd = [
            "ffmpeg",
            "-i",
            str(audio_file),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-",
        ]
        result = subprocess.run(decode_cmd, capture_output=True, check=True)
        y = np.frombuffer(result.stdout, dtype=np.float32)
        duration = len(y) / sr
        print(f"Loaded {duration:.1f}s audio ({len(y)} samples at {sr}Hz)")

        # Calculate number of frames needed
        n_frames = int(duration * fps)

        # Convert hex color to BGR (OpenCV uses BGR channel ordering)
        hex_color = dot_color.lstrip("#")
        rgb_color = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        bgr_color = rgb_color[::-1]

        # Pre-calculate colors: full brightness and half brightness (works on black bg)
        color_full = bgr_color
        color_half = tuple(c // 2 for c in bgr_color)

        # Pre-calculate radius and other constants
        radius = (dot_size + 1) // 2
        center_y = height // 2
        max_viz_height = int(height * max_height / 100)

        # Pre-calculate x positions
        n_dots = width // (dot_size + dot_spacing)
        x_positions = np.arange(n_dots) * (dot_size + dot_spacing) + dot_size // 2

        # Pre-calculate amplitudes using memory-efficient approach
        samples_per_frame = len(y) // n_frames
        chunk_size = samples_per_frame // n_dots
        # Only keep what we need, discard excess samples
        usable_samples = n_frames * n_dots * chunk_size
        y_trimmed = np.abs(y[:usable_samples]).reshape(n_frames, n_dots, chunk_size)
        amplitudes = y_trimmed.mean(axis=2)
        del y, y_trimmed  # Free audio data immediately
        max_amp = amplitudes.max(axis=1, keepdims=True)
        max_amp[max_amp == 0] = 1
        amplitudes = amplitudes / max_amp

        # Initialize frame buffer
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Set up ffmpeg pipe for streaming frames directly to encoder
        output_path = "/tmp/output.mp4"
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            # Video input: raw frames from stdin
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "bgr24",
            "-r",
            str(fps),
            "-i",
            "-",
            # Audio input: original audio file
            "-i",
            str(audio_file),
            # Video encoding — CRF 18 is visually lossless. Combined with
            # tune stillimage, this is well-suited for synthetic graphics
            # (colored dots on black).
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-tune",
            "stillimage",
            "-pix_fmt",
            "yuv420p",
            # Audio encoding
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            # Stop at shortest stream (video may be slightly shorter than audio)
            "-shortest",
            "-threads",
            "4",
            output_path,
        ]
        stderr_log = open("/tmp/ffmpeg_encode.log", "w")
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=stderr_log,
        )

        print(f"Generating and encoding {n_frames} frames...")
        for frame_idx in range(n_frames):
            if frame_idx % 500 == 0 and frame_idx > 0:
                print(
                    f"  Progress: {frame_idx}/{n_frames} ({100 * frame_idx / n_frames:.1f}%)"
                )

            self.frame.fill(0)  # Clear frame
            frame_amplitudes = amplitudes[frame_idx]

            # Calculate all y positions for symmetric dots
            y_offsets = np.minimum(
                frame_amplitudes * max_viz_height // 2, max_viz_height // 2
            )
            n_symmetric_dots = (y_offsets // (dot_size + dot_spacing)).astype(int) + 1

            # Draw center dots directly on frame
            mask = np.random.random(len(x_positions)) < 0.5
            for x in x_positions[mask]:
                cv2.circle(self.frame, (int(x), center_y), radius, color_full, -1)
            for x in x_positions[~mask]:
                cv2.circle(self.frame, (int(x), center_y), radius, color_half, -1)

            # Draw symmetric dots for each level
            max_dots = int(n_symmetric_dots.max())
            for j in range(1, max_dots):
                y_pos = j * (dot_size + dot_spacing)
                valid_dots = j < n_symmetric_dots

                if np.any(valid_dots):
                    x_valid = x_positions[valid_dots]
                    self.draw_symmetric_dots(
                        self.frame,
                        x_valid,
                        np.full_like(x_valid, center_y + y_pos),
                        np.full_like(x_valid, center_y - y_pos),
                        radius,
                        color_full,
                        color_half,
                    )

            # Stream frame directly to ffmpeg
            ffmpeg_proc.stdin.write(self.frame.tobytes())

        # Finalize encoding
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        stderr_log.close()

        if ffmpeg_proc.returncode != 0:
            with open("/tmp/ffmpeg_encode.log") as f:
                stderr = f.read()
            raise RuntimeError(f"ffmpeg encoding failed: {stderr}")

        print("Done!")
        return Path(output_path)
