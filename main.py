import subprocess
import os
import json
import sys
from multiprocessing import Pool, freeze_support
import shutil
from functools import partial
import hashlib
from typing import List, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)


def run_ffmpeg(cmd: List[str], log_file: Path) -> subprocess.CompletedProcess:
    """Run ffmpeg command and handle output redirection."""
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, check=True)
        return result
    except subprocess.CalledProcessError as e:
        # If there's an error, read and log the ffmpeg output
        with open(log_file, 'r') as f:
            ffmpeg_output = f.read()
        logging.error(f"FFmpeg output:\n{ffmpeg_output}")
        raise e


class ProcessingParams:
    def __init__(
            self,
            english_only: bool = True,
            processes: int = 4,
            recode: Optional[str] = None,
            output_dir: str = "out"
    ):
        self.english_only = english_only
        self.processes = processes
        self.recode = recode
        self.output_dir = output_dir
        logging.info(
            f"Initialized processing parameters: english_only={english_only}, processes={processes}, recode={recode}")


def get_stream_info(input_file: str) -> List[dict]:
    """Get information about all streams in the file using ffprobe."""
    logging.info(f"Getting stream information for: {input_file}")
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        input_file
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        if 'streams' not in data:
            raise RuntimeError("No stream information found in ffprobe output")

        streams = []
        for stream in data['streams']:
            stream_info = {
                'codec_type': stream.get('codec_type', ''),
                'codec_name': stream.get('codec_name', ''),
                'channels': stream.get('channels', 0),
                'bit_rate': stream.get('bit_rate', '0'),
                'tags': stream.get('tags', {})
            }
            streams.append(stream_info)

        if not streams:
            raise RuntimeError("No valid streams found in the file")

        logging.info(f"Found {len(streams)} streams in {input_file}")
        for stream in streams:
            logging.debug(f"Stream: {stream}")

        return streams

    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get stream info: {e}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse ffprobe output: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise


def extract_tracks(mkv_file: str, params: ProcessingParams) -> Tuple[
    List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Extract video, audio, and subtitle tracks using ffmpeg."""
    logging.info(f"Starting track extraction for: {mkv_file}")
    if not os.path.isfile(mkv_file):
        raise FileNotFoundError(f"No such file: '{mkv_file}'")

    output_dir = Path(params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    streams = get_stream_info(mkv_file)
    video, audio, subtitle = [], [], []

    sha3_hash = hashlib.sha3_512()
    sha3_hash.update(mkv_file.encode('utf-8'))
    hash_hex = sha3_hash.hexdigest()

    # Check if input is AVI or needs encoding due to recode parameter
    is_avi = Path(mkv_file).suffix.lower() == '.avi'
    needs_encoding = is_avi or (params.recode is not None and is_avi)  # Only encode AVI files here

    # Get original video bitrate if available
    original_bitrate = None
    for stream in streams:
        if stream['codec_type'] == 'video' and stream.get('bit_rate'):
            try:
                original_bitrate = int(stream['bit_rate'])
                logging.info(f"Original video bitrate: {original_bitrate} b/s")
            except (ValueError, TypeError):
                pass
            break

    # Check if any audio streams have language tags
    has_language_tags = any(
        stream.get('tags', {}).get('language') is not None
        for stream in streams
        if stream['codec_type'] == 'audio'
    )

    for idx, stream in enumerate(streams):
        stream_type = stream['codec_type']
        codec_name = stream['codec_name'].lower()
        channels = stream.get('channels', 0)
        lang = stream.get('tags', {}).get('language', 'und')

        # Define output file for this stream
        output_file = output_dir / f'{hash_hex}_{stream_type}_track_{idx}.mkv'
        logging.info(f"Extracting {stream_type} track {idx} ({lang}) to {output_file}")

        # Process each stream type
        if stream_type == 'video':
            video.append((str(output_file), lang))
            if needs_encoding:
                # Re-encode video to H.264 using NVIDIA GPU
                logging.info(f"Re-encoding video track {idx} to H.264 using NVIDIA GPU")
                cmd = [
                    'ffmpeg',
                    '-i', mkv_file,
                    '-map', f'0:{idx}',
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p4',  # Balanced preset for NVENC
                    '-rc', 'vbr',  # Variable bitrate mode
                ]

                # Use recode parameter if provided, otherwise use original bitrate
                if params.recode:
                    # Remove 'K' suffix and convert to integer
                    bitrate_value = int(params.recode[:-1])
                    target_bitrate = f"{bitrate_value}K"
                    maxrate = f"{bitrate_value * 2}K"
                    bufsize = f"{bitrate_value * 4}K"
                    cmd.extend([
                        '-b:v', target_bitrate,
                        '-maxrate', maxrate,
                        '-bufsize', bufsize,
                    ])
                elif original_bitrate:
                    # Use original bitrate if available
                    cmd.extend([
                        '-b:v', f'{original_bitrate}',
                        '-maxrate', f'{original_bitrate * 2}',
                        '-bufsize', f'{original_bitrate * 4}',
                    ])
                else:
                    # Fallback to quality-based encoding
                    cmd.extend([
                        '-cq', '23',  # Quality level (lower = better quality)
                        '-qmin', '0',  # Minimum QP value
                        '-qmax', '51',  # Maximum QP value
                        '-maxrate', '2M',  # Maximum bitrate
                        '-bufsize', '4M',  # Buffer size
                    ])

                cmd.extend(['-y', str(output_file)])
            else:
                cmd = [
                    'ffmpeg',
                    '-i', mkv_file,
                    '-map', f'0:{idx}',
                    '-c', 'copy',
                    '-y',
                    str(output_file)
                ]
        elif stream_type == 'audio':
            if not has_language_tags or not params.english_only or lang == 'eng':
                audio.append((str(output_file), lang))
                # Check if audio is DTS/DTS-HD and needs conversion
                if codec_name in ['dts', 'dtshd', 'dts-hd', 'dtshd_ma']:
                    logging.info(f"Converting DTS audio track {idx} to AAC with {channels} channels")
                    cmd = [
                        'ffmpeg',
                        '-i', mkv_file,
                        '-map', f'0:{idx}',
                        '-c:a', 'aac',
                        '-b:a', '0',  # Use maximum bitrate
                        '-ac', str(channels),  # Maintain original channel configuration
                        '-y',
                        str(output_file)
                    ]
                else:
                    cmd = [
                        'ffmpeg',
                        '-i', mkv_file,
                        '-map', f'0:{idx}',
                        '-c', 'copy',
                        '-y',
                        str(output_file)
                    ]
        elif stream_type == 'subtitle':
            if not params.english_only or lang == 'eng':
                subtitle.append((str(output_file), lang))
                cmd = [
                    'ffmpeg',
                    '-i', mkv_file,
                    '-map', f'0:{idx}',
                    '-c', 'copy',
                    '-y',
                    str(output_file)
                ]

        try:
            log_file = logs_dir / f"ffmpeg_extract_{hash_hex}_{stream_type}_{idx}.log"
            run_ffmpeg(cmd, log_file)
            logging.debug(f"Successfully extracted {stream_type} track {idx}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to extract {stream_type} track {idx}: {e}")
            raise

    logging.info(f"Extraction complete: {len(video)} video, {len(audio)} audio, {len(subtitle)} subtitle tracks")
    return video, audio, subtitle


def merge_tracks(video: List[Tuple[str, str]], audio: List[Tuple[str, str]],
                 subtitle: List[Tuple[str, str]], destination: str) -> None:
    """Merge tracks using ffmpeg."""
    logging.info(f"Starting track merge to: {destination}")
    if not video:
        raise ValueError("No video tracks to merge")

    cmd = ['ffmpeg', '-y']

    # Add video input
    cmd.extend(['-i', video[0][0]])

    # Add audio inputs
    for audio_file, _ in audio:
        cmd.extend(['-i', audio_file])

    # Add subtitle inputs
    for sub_file, _ in subtitle:
        cmd.extend(['-i', sub_file])

    # Map all streams to output
    cmd.extend(['-map', '0:v:0'])
    for i in range(len(audio)):
        cmd.extend(['-map', f'{i + 1}:a:0'])
    for i in range(len(subtitle)):
        cmd.extend(['-map', f'{i + 1 + len(audio)}:s:0'])

    # Set metadata for each stream
    for i, (_, lang) in enumerate(video):
        cmd.extend(['-metadata:s:v:0', f'language={lang}'])
    for i, (_, lang) in enumerate(audio):
        cmd.extend(['-metadata:s:a:' + str(i), f'language={lang}'])
    for i, (_, lang) in enumerate(subtitle):
        cmd.extend(['-metadata:s:s:' + str(i), f'language={lang}'])

    cmd.extend(['-c', 'copy', destination])

    logging.debug(f"Running ffmpeg command: {' '.join(cmd)}")
    try:
        hash_hex = hashlib.sha3_512()
        hash_hex.update(destination.encode('utf-8'))
        log_file = logs_dir / f"ffmpeg_merge_{hash_hex.hexdigest()}.log"
        run_ffmpeg(cmd, log_file)
        logging.info(f"Successfully merged tracks to {destination}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to merge tracks: {e}")
        raise


def reduce_bitrate(input_file: Tuple[str, str], bitrate: str = "700K") -> Tuple[str, str]:
    """Reduce the bitrate of a video file using ffmpeg."""
    logging.info(f"Reducing bitrate for: {input_file[0]} to {bitrate}")
    input_path = Path(input_file[0])
    output_file = input_path.parent / f"{input_path.stem}_encoded.mp4"

    # Get original video bitrate
    streams = get_stream_info(str(input_path))
    original_bitrate = None
    for stream in streams:
        if stream['codec_type'] == 'video' and stream.get('bit_rate'):
            try:
                original_bitrate = int(stream['bit_rate'])
                logging.info(f"Original video bitrate: {original_bitrate} b/s")
            except (ValueError, TypeError):
                pass
            break

    # Calculate maxrate and bufsize based on target bitrate
    maxrate = str(int(bitrate[:-1]) * 2) + bitrate[-1]  # Double the target bitrate
    bufsize = str(int(bitrate[:-1]) * 4) + bitrate[-1]  # Four times the target bitrate

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-c:v", "h264_nvenc",
        "-preset", "p4",  # Balanced preset
        "-rc", "vbr",  # Variable bitrate mode
        "-b:v", bitrate,
        "-maxrate", maxrate,
        "-bufsize", bufsize,
        "-qmin", "0",  # Minimum QP value
        "-qmax", "51",  # Maximum QP value
        "-c:a", "copy",
        "-y",
        str(output_file)
    ]

    try:
        hash_hex = hashlib.sha3_512()
        hash_hex.update(str(input_path).encode('utf-8'))
        log_file = logs_dir / f"ffmpeg_reduce_{hash_hex.hexdigest()}.log"
        run_ffmpeg(cmd, log_file)
        logging.info(f"Successfully reduced bitrate to {bitrate}")
        return (str(output_file), input_file[1])
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to reduce bitrate: {e}")
        raise


def delete_tracks(video: List[Tuple[str, str]], audio: List[Tuple[str, str]], subtitle: List[Tuple[str, str]]) -> None:
    """Delete temporary track files."""
    logging.debug("Cleaning up temporary files")
    for file_path, _ in video + audio + subtitle:
        try:
            Path(file_path).unlink()
            logging.debug(f"Deleted temporary file: {file_path}")
        except FileNotFoundError:
            logging.debug(f"Temporary file already deleted: {file_path}")


def get_supported_formats() -> List[str]:
    """Get list of supported video formats from ffmpeg."""
    try:
        result = subprocess.run(['ffmpeg', '-formats'], capture_output=True, text=True)
        formats = []
        for line in result.stdout.split('\n'):
            if ' D ' in line and 'video' in line.lower():
                format_name = line.split()[1]
                formats.append(format_name)
        return formats
    except Exception as e:
        logging.error(f"Failed to get supported formats: {e}")
        # Return common video formats as fallback
        return ['mkv', 'mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'm4v']


def is_video_file(file_path: Path) -> bool:
    """Check if file is a video file using ffprobe."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return 'video' in result.stdout
    except:
        return False


def worker(x: str, local_params: ProcessingParams) -> None:
    """Process a single video file."""
    logging.info(f"Starting processing of: {x}")
    video, audio, subtitle = [], [], []
    try:
        video, audio, subtitle = extract_tracks(x, local_params)
        # Only reduce bitrate if recode parameter is provided and it's not an AVI file
        if local_params.recode and not x.lower().endswith('.avi'):
            logging.info(f"Recoding video track for: {x}")
            video[0] = reduce_bitrate(video[0], local_params.recode)
        # Always output as MKV for consistency
        output_file = str(Path(x).with_suffix('.mkv'))
        merge_tracks(video, audio, subtitle, output_file)
        logging.info(f"Successfully processed: {x}")
    except Exception as e:
        logging.error(f"Failed to process {x}: {e}")
        raise
    finally:
        delete_tracks(video, audio, subtitle)


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <directory> [bitrate]")
        sys.exit(1)

    dir_to_process = Path(sys.argv[1])
    if not dir_to_process.is_dir():
        logging.error(f"Error: '{dir_to_process}' is not a directory")
        sys.exit(1)

    params = ProcessingParams(
        recode=f'{sys.argv[2]}K' if len(sys.argv) > 2 else None
    )

    try:
        # Get all files and filter video files
        files = []
        for f in dir_to_process.rglob("*"):
            if f.is_file() and is_video_file(f):
                files.append(str(f))

        if not files:
            logging.warning(f"No video files found in '{dir_to_process}'")
            sys.exit(0)

        logging.info(f"Found {len(files)} video files to process")

        freeze_support()
        with Pool(params.processes) as pool:
            pool.map(partial(worker, local_params=params), files)

    except KeyboardInterrupt:
        logging.warning("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)
    finally:
        try:
            shutil.rmtree(params.output_dir, ignore_errors=True)
            logging.info("Cleaned up output directory")
        except Exception as e:
            logging.warning(f"Failed to clean up output directory: {e}")


if __name__ == "__main__":
    main()
