"""
ATC Audio Transcription Pipeline (Lightweight crispasr.exe Version)
Pipeline with Noise Removal (Filter + Spectral NR) and crispasr.exe Wrapper
"""

import os
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path
import time
import argparse
import subprocess
import re
import datetime


class AudioTranscriptionPipeline:
    """
    Complete audio transcription pipeline for noisy ATC recordings
    Includes: Filter + Noise Removal -> crispasr.exe Wrapper
    """
    
    def __init__(self, 
                 exe_path: str = r"bin\cuda\crispasr.exe",
                 model_path: str = r"models\ggufs\speech-model.gguf",
                 sample_rate: int = 16000,
                 use_gpu: bool = True,
                 # Keep other parameters for compatibility, but ignored
                 whisper_model: Optional[str] = None,
                 parakeet_model_dir: Optional[str] = None,
                 backend: Optional[str] = None):
        """
        Initialize the transcription pipeline
        
        Args:
            exe_path: Path to the crispasr.exe executable
            model_path: Path to the GGUF model
            sample_rate: Target sample rate
            use_gpu: Whether to use GPU/CUDA
        """
        self.sample_rate = sample_rate
        self.exe_path = exe_path
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.backend = "crispasr"
        
        # Backward compatibility check
        if whisper_model and os.path.exists(whisper_model) and whisper_model.endswith(".gguf"):
            self.model_path = whisper_model
            
        print(f"Initializing crispasr.exe pipeline...")
        print(f"  Executable: {self.exe_path}")
        print(f"  Model: {self.model_path}")
        
    def _load_vad_model(self):
        """Deprecated: VAD is handled internally by crispasr.exe"""
        return None, (None, None, None, None, None)
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample to target sample rate
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        print(f"Loading audio from: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return audio, sr
    
    def apply_vad(self, 
                  audio: np.ndarray, 
                  threshold: float = 0.5,
                  min_speech_duration_ms: int = 250,
                  min_silence_duration_ms: int = 100) -> List[Dict]:
        """Deprecated: VAD is handled internally by crispasr.exe"""
        print("VAD is now handled internally by crispasr.exe. Returning empty segment list.")
        return []
    
    def remove_noise(self, 
                     audio: np.ndarray,
                     stationary: bool = True,
                     prop_decrease: float = 1.0) -> np.ndarray:
        """
        Remove noise from audio using spectral gating
        
        Args:
            audio: Input audio signal
            stationary: Use stationary or non-stationary noise reduction
            prop_decrease: Proportion of noise to reduce (0-1)
            
        Returns:
            Denoised audio signal
        """
        print("Removing noise...")
        
        try:
            import noisereduce as nr
            
            # Apply noise reduction
            denoised_audio = nr.reduce_noise(
                y=audio,
                sr=self.sample_rate,
                stationary=stationary,
                prop_decrease=prop_decrease
            )
            
            return denoised_audio
        except ImportError:
            print("Warning: noisereduce not installed. Skipping noise removal.")
            print("Install with: pip install noisereduce")
            return audio

    def apply_deepfilternet(self, audio: np.ndarray) -> np.ndarray:
        """
        Deprecated/Not recommended under lightweight env, but kept for compatibility.
        """
        print("Skipping DeepFilterNet (unsupported in lightweight env).")
        return audio

    def apply_bandpass_filter(self, 
                            audio: np.ndarray, 
                            lowcut: float = 300.0, 
                            highcut: float = 3400.0, 
                            order: int = 5) -> np.ndarray:
        """
        Apply Bandpass Filter (Butterworth) to keep frequencies relevant for human speech.
        """
        try:
            from scipy.signal import butter, lfilter
            
            nyq = 0.5 * self.sample_rate
            
            # Determine filter type based on bounds
            use_low = (lowcut > 0)
            use_high = (highcut < nyq)
            
            if not use_low and not use_high:
                print("Skipping filter: both lowcut and highcut are outside bounds.")
                return audio
            
            if use_low and use_high:
                print(f"Applying Bandpass Filter ({lowcut}Hz - {highcut}Hz)...")
                low = lowcut / nyq
                high = highcut / nyq
                b, a = butter(order, [low, high], btype='band')
            elif use_low:
                print(f"Applying High-pass Filter (>{lowcut}Hz)...")
                low = lowcut / nyq
                b, a = butter(order, low, btype='high')
            else:
                print(f"Applying Low-pass Filter (<{highcut}Hz)...")
                high = highcut / nyq
                b, a = butter(order, high, btype='low')
                
            filtered_audio = lfilter(b, a, audio)
            return filtered_audio
        except ImportError:
            print("Warning: scipy not installed. Skipping bandpass filter.")
            return audio
        except Exception as e:
            print(f"Error applying bandpass filter: {e}")
            return audio

    def extract_speech_segments(self, 
                                audio: np.ndarray,
                                speech_timestamps: List[Dict]) -> np.ndarray:
        """Deprecated: VAD is handled internally by crispasr.exe"""
        return audio
    
    def transcribe(self, 
                   audio: Union[np.ndarray, str],
                   language: str = "en",
                   task: str = "transcribe") -> Dict:
        """
        Transcribe audio using crispasr.exe subprocess wrapper
        
        Args:
            audio: Numpy array of audio data, or path to audio file
            language: Language code (default: 'en')
            task: Action (ignored)
            
        Returns:
            Transcription result dictionary
        """
        import tempfile
        import re
        
        temp_wav_path = None
        if isinstance(audio, np.ndarray):
            # Write numpy array to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_wav_path = tmp.name
            sf.write(temp_wav_path, audio, self.sample_rate)
            audio_file_path = temp_wav_path
        else:
            audio_file_path = audio
            
        try:
            # Build command with optimizations: beam search, forced English language, and domain-specific hotwords
            hotwords = (
                "ils,dme,localizer,runway,heading,knots,feet,thousand,"
                "altitude,approach,climb,descend,contact,tower,niner"
            )
            cmd = [
                self.exe_path,
                "-m", self.model_path,
                "-f", audio_file_path,
                "--vad",
                "--flush-after", "1",
                "-osrt",
                "-bs", "5",         # Enable beam search size 5
                "-l", "en",         # Force English language to prevent auto-detect errors
                "--hotwords", hotwords # Bias callsigns and waypoints
            ]
            
            print(f"Running backend command: {' '.join(cmd)}")
            
            # Execute subprocess
            process_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                check=False
            )
            
            stdout = process_result.stdout
            stderr = process_result.stderr
            
            if process_result.returncode != 0:
                print(f"Warning: crispasr.exe exited with return code {process_result.returncode}")
                print(f"Stderr:\n{stderr}")
                
            # Try to read the SRT file if generated
            srt_path_1 = audio_file_path + ".srt"
            srt_path_2 = os.path.splitext(audio_file_path)[0] + ".srt"
            
            segments = []
            srt_found = False
            
            for srt_path in [srt_path_1, srt_path_2]:
                if os.path.exists(srt_path):
                    print(f"Parsing timestamps from generated SRT: {srt_path}")
                    segments = self._parse_srt_file(srt_path)
                    try:
                        os.remove(srt_path)
                    except Exception as e:
                        print(f"Warning: Could not remove SRT file {srt_path}: {e}")
                    if segments:
                        srt_found = True
                        break
            
            # If no SRT file on disk, but stdout contains SRT formatting
            if not srt_found and "-->" in stdout:
                print("Parsing SRT content directly from stdout...")
                segments = self._parse_srt_content(stdout)
                if segments:
                    srt_found = True
                    
            if srt_found:
                text_lines = [seg["text"] for seg in segments]
                full_text = " ".join(text_lines).strip()
            else:
                # Parse text lines from stdout
                text_lines = []
                stdout_lines = stdout.splitlines()
                for line in stdout_lines:
                    line_strip = line.strip()
                    if line_strip and not self._is_log_line(line_strip):
                        # Clean timestamp prefix if present in the line (e.g. from stdout)
                        cleaned_line = re.sub(r'^\[\d{2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,\.]\d{3}\]\s*', '', line_strip)
                        text_lines.append(cleaned_line.strip())
                
                full_text = " ".join(text_lines).strip()
                
                # Fallback 1: If SRT was not generated/found but VAD info is in stderr logs,
                # parse VAD segments from stderr and pair them with stdout text lines
                print("No SRT file found. Attempting to parse VAD segments from stderr logs...")
                vad_segments = self._parse_vad_from_stderr(stderr)
                
                if vad_segments and text_lines:
                    print(f"Found {len(vad_segments)} VAD segments and {len(text_lines)} transcribed text lines.")
                    # Pair them up
                    for i in range(min(len(vad_segments), len(text_lines))):
                        segments.append({
                            "start": vad_segments[i]["start"],
                            "end": vad_segments[i]["end"],
                            "text": text_lines[i]
                        })
                    # If we have leftover text lines, combine them into the last segment
                    if len(text_lines) > len(vad_segments):
                        extra_text = " ".join(text_lines[len(vad_segments):])
                        if segments:
                            segments[-1]["text"] += " " + extra_text
                elif vad_segments and full_text:
                    segments.append({
                        "start": vad_segments[0]["start"],
                        "end": vad_segments[-1]["end"],
                        "text": full_text
                    })
            
            # Fallback 2: General fallback (single global segment)
            if not segments and full_text:
                segments.append({
                    "start": 0.0,
                    "end": 0.0,
                    "text": full_text
                })
                
            # Split long segments to improve readability
            segments = self._split_long_segments(segments, max_words=10)
                
            return {
                "text": full_text,
                "segments": segments,
                "language": language
            }
            
        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except Exception as e:
                    print(f"Warning: Could not remove temp file {temp_wav_path}: {e}")

    def _parse_srt_file(self, srt_path: str) -> List[Dict]:
        try:
            with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return self._parse_srt_content(content)
        except Exception as e:
            print(f"Warning: Failed to parse SRT file {srt_path}: {e}")
            return []

    def _parse_srt_content(self, content: str) -> List[Dict]:
        try:
            segments = []
            content = content.replace('\r\n', '\n')
            
            # Match SRT blocks
            srt_pattern = re.compile(
                r'(?:\d+)\n(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\n(.*?)(?=\n\s*\n|\n\d+\n|\Z)',
                re.DOTALL
            )
            
            matches = srt_pattern.findall(content)
            for start_str, end_str, text_content in matches:
                start_sec = self._parse_timestamp_to_seconds(start_str)
                end_sec = self._parse_timestamp_to_seconds(end_str)
                clean_lines = [line.strip() for line in text_content.strip().splitlines()]
                clean_text = " ".join([l for l in clean_lines if l]).strip()
                
                if clean_text:
                    segments.append({
                        "start": start_sec,
                        "end": end_sec,
                        "text": clean_text
                    })
            return segments
        except Exception as e:
            print(f"Warning: Failed to parse SRT content: {e}")
            return []

    def _split_long_segments(self, segments: List[Dict], max_words: int = 10) -> List[Dict]:
        new_segments = []
        for seg in segments:
            text = seg["text"].strip()
            words = text.split()
            if len(words) <= max_words:
                new_segments.append(seg)
                continue
                
            start = seg["start"]
            end = seg["end"]
            duration = end - start
            num_words = len(words)
            
            if duration <= 0:
                new_segments.append(seg)
                continue
                
            for i in range(0, num_words, max_words):
                chunk_words = words[i : i + max_words]
                chunk_text = " ".join(chunk_words)
                
                chunk_start = start + (i / num_words) * duration
                chunk_end = start + (min(i + max_words, num_words) / num_words) * duration
                
                new_segments.append({
                    "start": round(chunk_start, 2),
                    "end": round(chunk_end, 2),
                    "text": chunk_text
                })
        return new_segments

    def _parse_vad_from_stderr(self, stderr: str) -> List[Dict]:
        vad_segments = []
        pattern = re.compile(
            r'VAD segment\s*\d+:\s*start\s*=\s*([\d\.]+),\s*end\s*=\s*([\d\.]+)'
        )
        for line in stderr.splitlines():
            match = pattern.search(line)
            if match:
                start_val = float(match.group(1))
                end_val = float(match.group(2))
                vad_segments.append({
                    "start": start_val,
                    "end": end_val
                })
        return vad_segments

    def _parse_timestamp_to_seconds(self, ts_str: str) -> float:
        try:
            ts_str = ts_str.replace(',', '.')
            parts = ts_str.split(':')
            if len(parts) == 3:
                h, m, s = parts
                return float(h) * 3600 + float(m) * 60 + float(s)
            elif len(parts) == 2:
                m, s = parts
                return float(m) * 60 + float(s)
            return float(ts_str)
        except Exception:
            return 0.0

    def _is_log_line(self, line: str) -> bool:
        log_prefixes = [
            "system_info:", "whisper_init", "model", "main:", "whisper_", 
            "llama_", "ggml_", "load_", "warning:", "error:", "info:"
        ]
        line_lower = line.lower()
        for prefix in log_prefixes:
            if line_lower.startswith(prefix):
                return True
        if all(c in '-=_* \t' for c in line):
            return True
        return False
    
    def process_audio(self,
                      audio_path: str,
                      apply_vad: bool = True, # Ignored, handled by exe
                      apply_noise_removal: bool = True,
                      vad_threshold: float = 0.5,
                      noise_reduction: float = 0.75,
                      language: str = "en",
                      save_intermediate: bool = False,
                      output_dir: Optional[str] = None,
                      chunk_duration: float = 30.0,
                      overlap_duration: float = 5.0) -> Dict:
        """
        Complete pipeline: Load -> Filter & Denoise -> Save Temp -> Transcribe (crispasr.exe)
        """
        start_time = time.time()
        
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        # Generate timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filtered_audio = audio
        denoised_audio = audio
        
        filtered_full_path = None
        filtered_full_path_rel = None
        denoised_full_path = None
        denoised_full_path_rel = None
        intermediate_chunks = []
        
        # Apply Global Noise Removal (Bandpass + Spectral Gating)
        if apply_noise_removal:
            print("Applying global noise removal...")
            # 1. Bandpass Filter (Widened: 100Hz - 8000Hz)
            filtered_audio = self.apply_bandpass_filter(audio, lowcut=100.0, highcut=8000.0)
            
            if save_intermediate and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filtered_path = os.path.join(output_dir, f"filtered_full_{timestamp}.wav")
                sf.write(filtered_path, filtered_audio, self.sample_rate)
                filtered_full_path = os.path.abspath(filtered_path)
                filtered_full_path_rel = os.path.relpath(filtered_path).replace("\\", "/")
                print(f"Saved global filtered audio: {filtered_full_path}")
            
            # 2. Spectral Gating (noisereduce)
            denoised_audio = self.remove_noise(filtered_audio.copy(), prop_decrease=noise_reduction)
            
            if save_intermediate and output_dir:
                denoised_path = os.path.join(output_dir, f"denoised_full_{timestamp}.wav")
                sf.write(denoised_path, denoised_audio, self.sample_rate)
                denoised_full_path = os.path.abspath(denoised_path)
                denoised_full_path_rel = os.path.relpath(denoised_path).replace("\\", "/")
                print(f"Saved global denoised audio: {denoised_full_path}")
        
        # Save the filtered & denoised audio to a temp file for crispasr.exe
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_wav_path = tmp.name
            
        try:
            sf.write(temp_wav_path, denoised_audio, self.sample_rate)
            
            # Transcribe
            result = self.transcribe(temp_wav_path, language=language)
        finally:
            if os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except Exception as e:
                    print(f"Warning: Could not remove temp file {temp_wav_path}: {e}")
        
        # Save intermediate files in overlapping chunks
        if save_intermediate and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            step_duration_sec = chunk_duration - overlap_duration
            
            chunk_samples = int(chunk_duration * self.sample_rate)
            step_samples = int(step_duration_sec * self.sample_rate)
            total_samples = len(filtered_audio)
            
            chunk_idx = 0
            while True:
                start_sample = chunk_idx * step_samples
                end_sample = start_sample + chunk_samples
                
                if start_sample >= total_samples:
                    break
                    
                end_sample = min(end_sample, total_samples)
                chunk_num = chunk_idx + 1
                
                # Slice and save filtered chunk
                chunk_filtered = filtered_audio[start_sample:end_sample]
                filtered_chunk_path = os.path.join(output_dir, f"filtered_chunk_{chunk_num}_{timestamp}.wav")
                sf.write(filtered_chunk_path, chunk_filtered, self.sample_rate)
                
                # Slice and save denoised chunk
                chunk_denoised = denoised_audio[start_sample:end_sample]
                denoised_chunk_path = os.path.join(output_dir, f"denoised_chunk_{chunk_num}_{timestamp}.wav")
                sf.write(denoised_chunk_path, chunk_denoised, self.sample_rate)
                
                intermediate_chunks.append({
                    "chunk_num": chunk_num,
                    "start_time": round(start_sample / self.sample_rate, 2),
                    "end_time": round(end_sample / self.sample_rate, 2),
                    "filtered_path": os.path.abspath(filtered_chunk_path),
                    "filtered_path_rel": os.path.relpath(filtered_chunk_path).replace("\\", "/"),
                    "denoised_path": os.path.abspath(denoised_chunk_path),
                    "denoised_path_rel": os.path.relpath(denoised_chunk_path).replace("\\", "/")
                })
                
                if end_sample >= total_samples:
                    break
                    
                chunk_idx += 1
                
            print(f"Saved {chunk_idx + 1} intermediate overlapping 30-second filtered and denoised chunks in {output_dir}")
        
        # Prepare output
        output = {
            "text": result["text"],
            "segments": result.get("segments", []),
            "language": result["language"],
            "processing_time_seconds": time.time() - start_time,
            "speech_duration_seconds": len(audio) / self.sample_rate,
            "vad_enabled": True,
            "noise_removal_enabled": apply_noise_removal,
            "timestamp": timestamp,
            "filtered_full_path": filtered_full_path,
            "filtered_full_path_rel": filtered_full_path_rel,
            "denoised_full_path": denoised_full_path,
            "denoised_full_path_rel": denoised_full_path_rel,
            "intermediate_chunks": intermediate_chunks
        }
        
        print(f"\nTranscription complete in {time.time() - start_time:.2f}s")
        print(f"Transcribed text: {result['text'][:100]}...")
        return output

    def process_audio_steps(self,
                            audio_path: str,
                            apply_bandpass: bool = False,
                            apply_spectral_noise_reduction: bool = False,
                            apply_deepfilternet: bool = False,
                            apply_vad: bool = False,
                            vad_threshold: float = 0.5,
                            noise_reduction_prop: float = 0.75,
                            output_dir: str = "output/testing") -> Dict:
        """
        Run the pipeline step-by-step for testing and return paths to the intermediate files.
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        # 1. Original
        audio, sr = self.load_audio(audio_path)
        orig_path = os.path.join(output_dir, "step_0_original.wav")
        sf.write(orig_path, audio, sr)
        results['original'] = orig_path
        
        current_audio = audio
        
        # 2. Bandpass Filter
        if apply_bandpass:
            current_audio = self.apply_bandpass_filter(current_audio, lowcut=100.0, highcut=8000.0)
            bp_path = os.path.join(output_dir, "step_1_bandpass.wav")
            sf.write(bp_path, current_audio, sr)
            results['bandpass'] = bp_path
            
        # 3. Spectral Noise Reduction
        if apply_spectral_noise_reduction:
            current_audio = self.remove_noise(current_audio, prop_decrease=noise_reduction_prop)
            snr_path = os.path.join(output_dir, "step_2_spectral_nr.wav")
            sf.write(snr_path, current_audio, sr)
            results['spectral_nr'] = snr_path
            
        # 4. DeepFilterNet (skipped)
        if apply_deepfilternet:
            print("Skipping DeepFilterNet step.")
            
        # Save final preprocessed
        final_processed_path = os.path.join(output_dir, "step_final_processed.wav")
        sf.write(final_processed_path, current_audio, sr)
        results['final_processed'] = final_processed_path
        
        # 5. Transcribe final processed audio
        transcription_result = self.transcribe(final_processed_path)
        results['transcription'] = transcription_result.get('text', '')
            
        return results
    
    def save_transcription(self, result: Dict, output_path: str):
        """Save transcription result to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved transcription to: {output_path}")


def main():
    """CLI usage of the pipeline"""
    parser = argparse.ArgumentParser(description="ATC Audio Transcription Pipeline (Lightweight crispasr.exe version)")
    parser.add_argument("audio_file", help="Path to input audio file")
    parser.add_argument("--model", default=r"models\ggufs\speech-model.gguf", help="Path to local GGUF model")
    parser.add_argument("--exe_path", default=r"bin\cuda\crispasr.exe", help="Path to crispasr.exe")
    parser.add_argument("--output_dir", default="output", help="Directory to save output")
    parser.add_argument("--no_noise_removal", action="store_true", help="Disable noise removal")
    parser.add_argument("--noise_reduction", type=float, default=0.75, help="Noise reduction amount (default: 0.75)")
    parser.add_argument("--language", default="en", help="Language code")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AudioTranscriptionPipeline(
        exe_path=args.exe_path,
        model_path=args.model,
        sample_rate=16000
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    result = pipeline.process_audio(
        audio_path=args.audio_file,
        apply_noise_removal=not args.no_noise_removal,
        noise_reduction=args.noise_reduction,
        language=args.language,
        save_intermediate=True,
        output_dir=args.output_dir
    )
    
    # Save transcription
    timestamp = result.get("timestamp", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_path = os.path.join(args.output_dir, f"{Path(args.audio_file).stem}_transcription_{timestamp}.json")
    pipeline.save_transcription(result, output_path)
    
    return result


if __name__ == "__main__":
    main()