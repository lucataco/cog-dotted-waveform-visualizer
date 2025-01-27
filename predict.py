from cog import BasePredictor, Input, Path
import cv2
import librosa
import numpy as np
from moviepy.editor import ImageSequenceClip, AudioFileClip

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Pre-allocate numpy arrays for better performance
        self.overlay = None
        self.frame = None
    
    def draw_symmetric_dots_vectorized(self, frame, x_coords, y1_coords, y2_coords, radius, color_full, color_half):
        """Vectorized version of drawing dots"""
        h, w = frame.shape[:2]
        y1_coords = np.clip(y1_coords, radius, h-radius)
        y2_coords = np.clip(y2_coords, radius, h-radius)
        
        # Create masks for full and half opacity
        mask = np.random.random(len(x_coords)) < 0.5
        self.overlay.fill(0)  # Clear overlay
        
        # Draw dots with full opacity
        full_mask = mask
        if np.any(full_mask):
            for x, y1, y2 in zip(x_coords[full_mask], y1_coords[full_mask], y2_coords[full_mask]):
                cv2.circle(self.overlay, (int(x), int(y1)), radius, color_full[:3], -1, cv2.LINE_AA)
                cv2.circle(self.overlay, (int(x), int(y2)), radius, color_full[:3], -1, cv2.LINE_AA)
            cv2.addWeighted(self.overlay, 1.0, frame, 1.0, 0, frame)
        
        # Draw dots with half opacity
        self.overlay.fill(0)  # Clear overlay
        half_mask = ~mask
        if np.any(half_mask):
            for x, y1, y2 in zip(x_coords[half_mask], y1_coords[half_mask], y2_coords[half_mask]):
                cv2.circle(self.overlay, (int(x), int(y1)), radius, color_half[:3], -1, cv2.LINE_AA)
                cv2.circle(self.overlay, (int(x), int(y2)), radius, color_half[:3], -1, cv2.LINE_AA)
            cv2.addWeighted(self.overlay, 0.5, frame, 1.0, 0, frame)

    def predict(
        self,
        audio_file: Path = Input(description="Input audio file"),
        dot_size: int = Input(description="Size of dots in pixels", default=6),
        dot_spacing: int = Input(description="Spacing between dots in pixels", default=6),
        height: int = Input(description="Height of the output video in pixels", default=720, ge=100, le=1280),
        width: int = Input(description="Width of the output video in pixels", default=1280, ge=100, le=1280),
        max_height: int = Input(description="Maximum height of visualization as a percentage", default=30, ge=5, le=100),
        dot_color: str = Input(description="Dot color in hex format", default="#00FFFF"),
        fps: int = Input(description="Frames per second", default=10, ge=1, le=30),
    ) -> Path:
        """Run a single prediction on the model"""
        
        print("Loading audio file...")
        y, sr = librosa.load(str(audio_file))
        print("Finished loading audio file")
        
        # Calculate number of frames needed
        duration = len(y) / sr
        n_frames = int(duration * fps)
        
        # Convert hex color to RGB
        hex_color = dot_color.lstrip('#')
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Pre-calculate colors with different opacities
        color_full = (*rgb_color, 255)
        color_half = (*rgb_color, 128)
        
        # Pre-calculate radius and other constants
        radius = (dot_size + 1) // 2
        center_y = height // 2
        max_viz_height = int(height * max_height / 100)
        
        # Pre-calculate x positions
        n_dots = width // (dot_size + dot_spacing)
        x_positions = np.arange(n_dots) * (dot_size + dot_spacing) + dot_size // 2
        
        # Pre-calculate amplitudes
        samples_per_frame = len(y) // n_frames
        frame_samples = y[:n_frames * samples_per_frame].reshape(n_frames, samples_per_frame)
        chunk_size = samples_per_frame // n_dots
        frame_chunks = frame_samples[:, :chunk_size * n_dots].reshape(n_frames, n_dots, chunk_size)
        amplitudes = np.abs(frame_chunks).mean(axis=2)
        max_amp = amplitudes.max(axis=1, keepdims=True)
        max_amp[max_amp == 0] = 1
        amplitudes = amplitudes / max_amp
        
        # Initialize frame buffer and overlay
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.overlay = np.zeros_like(self.frame)
        frames = []
        
        print("Generating frames...")
        for frame_idx in range(n_frames):
            self.frame.fill(0)  # Clear frame
            frame_amplitudes = amplitudes[frame_idx]
            
            # Calculate all y positions for symmetric dots
            y_offsets = np.minimum(frame_amplitudes * max_viz_height // 2, max_viz_height // 2)
            n_symmetric_dots = (y_offsets // (dot_size + dot_spacing)).astype(int) + 1
            
            # Draw center dots
            self.overlay.fill(0)
            mask = np.random.random(len(x_positions)) < 0.5
            
            # Draw full opacity center dots
            for x in x_positions[mask]:
                cv2.circle(self.overlay, (int(x), center_y), radius, color_full[:3], -1, cv2.LINE_AA)
            cv2.addWeighted(self.overlay, 1.0, self.frame, 1.0, 0, self.frame)
            
            # Draw half opacity center dots
            self.overlay.fill(0)
            for x in x_positions[~mask]:
                cv2.circle(self.overlay, (int(x), center_y), radius, color_half[:3], -1, cv2.LINE_AA)
            cv2.addWeighted(self.overlay, 0.5, self.frame, 1.0, 0, self.frame)
            
            # Draw symmetric dots for each level
            max_dots = int(n_symmetric_dots.max())
            for j in range(1, max_dots):
                y_pos = j * (dot_size + dot_spacing)
                valid_dots = j < n_symmetric_dots
                
                if np.any(valid_dots):
                    x_valid = x_positions[valid_dots]
                    self.draw_symmetric_dots_vectorized(
                        self.frame, x_valid,
                        np.full_like(x_valid, center_y + y_pos),
                        np.full_like(x_valid, center_y - y_pos),
                        radius, color_full, color_half
                    )
            
            frames.append(self.frame.copy())
        
        print("Encoding video...")
        output_path = "/tmp/output.mp4"
        clip = ImageSequenceClip(frames, fps=fps)
        clip.audio = AudioFileClip(str(audio_file))
        clip.write_videofile(output_path, fps=fps, codec='libx264',
            preset='ultrafast',
            audio_codec='aac',
            threads=4,
            logger=None)
        
        return Path(output_path)
