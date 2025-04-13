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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)


def run_ffmpeg(cmd: List[str], log_file: Path) -> subprocess.CompletedProcess:
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, check=True)
        return result
    except subprocess.CalledProcessError as e:
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
    List[Tuple[str, str]],
    List[Tuple[str, str]],
    List[Tuple[str, str]]
]:
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

    is_avi = Path(mkv_file).suffix.lower() == '.avi'
    video_needs_encoding = is_avi or (params.recode is not None and is_avi)
    original_bitrate = None
    for stream in streams:
        if stream['codec_type'] == 'video' and stream.get('bit_rate'):
            try:
                original_bitrate = int(stream['bit_rate'])
                logging.info(f"Original video bitrate: {original_bitrate} b/s")
            except (ValueError, TypeError):
                pass
            break

    for idx, stream in enumerate(streams):
        stream_type = stream['codec_type']
        codec_name = stream['codec_name'].lower()
        channels = stream.get('channels', 0)
        lang = stream.get('tags', {}).get('language', 'und')
        output_file = output_dir / f'{hash_hex}_{stream_type}_track_{idx}.mkv'
        logging.info(f"Extracting {stream_type} track {idx} ({lang}) to {output_file}")
        cmd = []
        if stream_type == 'video':
            video.append((str(output_file), lang))
            if video_needs_encoding:
                logging.info(f"Re-encoding video track {idx} to H.264 using NVIDIA GPU")
                cmd = [
                    'ffmpeg',
                    '-i', mkv_file,
                    '-map', f'0:{idx}',
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p4',
                    '-rc', 'vbr',
                ]

                if params.recode:
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
                    cmd.extend([
                        '-b:v', f'{original_bitrate}',
                        '-maxrate', f'{original_bitrate * 2}',
                        '-bufsize', f'{original_bitrate * 4}',
                    ])
                else:
                    cmd.extend([
                        '-cq', '23',
                        '-qmin', '0',
                        '-qmax', '51',
                        '-maxrate', '2M',
                        '-bufsize', '4M',
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
            if params.english_only or lang == 'eng' or lang == '':
                audio.append((str(output_file), lang))
                if 'dts' in codec_name:
                    logging.info(f"Converting DTS audio track {idx} to E-AC3 with {channels} channels")
                    cmd = [
                        'ffmpeg',
                        '-i', mkv_file,
                        '-map', f'0:{idx}',
                        '-c:a', 'eac3',
                        '-b:a', '2000K',
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
            else:
                continue
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
    logging.info(f"Starting track merge to: {destination}")
    if not video:
        raise ValueError("No video tracks to merge")

    cmd = ['ffmpeg', '-y']

    cmd.extend(['-i', video[0][0]])

    for audio_file, _ in audio:
        cmd.extend(['-i', audio_file])

    for sub_file, _ in subtitle:
        cmd.extend(['-i', sub_file])

    cmd.extend(['-map', '0:v:0'])
    for i in range(len(audio)):
        cmd.extend(['-map', f'{i + 1}:a:0'])
    for i in range(len(subtitle)):
        cmd.extend(['-map', f'{i + 1 + len(audio)}:s:0'])

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


def change_bitrate(input_file: Tuple[str, str], bitrate: str = "700K") -> Tuple[str, str]:
    logging.info(f"Reducing bitrate for: {input_file[0]} to {bitrate}")
    input_path = Path(input_file[0])
    output_file = input_path.parent / f"{input_path.stem}_encoded.mp4"

    maxrate = str(int(bitrate[:-1]) * 2) + bitrate[-1]
    bufsize = str(int(bitrate[:-1]) * 4) + bitrate[-1]
    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-rc", "vbr",
        "-b:v", bitrate,
        "-maxrate", maxrate,
        "-bufsize", bufsize,
        "-qmin", "0",
        "-qmax", "51",
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
        return str(output_file), input_file[1]
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to reduce bitrate: {e}")
        raise


def delete_tracks(video: List[Tuple[str, str]], audio: List[Tuple[str, str]], subtitle: List[Tuple[str, str]]) -> None:
    logging.debug("Cleaning up temporary files")
    for file_path, _ in video + audio + subtitle:
        try:
            Path(file_path).unlink()
            logging.debug(f"Deleted temporary file: {file_path}")
        except FileNotFoundError:
            logging.debug(f"Temporary file already deleted: {file_path}")


def get_supported_formats() -> List[str]:
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
        return ['mkv', 'mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'm4v']


def is_video_file(file_path: Path) -> bool:
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
    logging.info(f"Starting processing of: {x}")
    video, audio, subtitle = [], [], []
    try:
        video, audio, subtitle = extract_tracks(x, local_params)
        if local_params.recode and not x.lower().endswith('.avi'):
            logging.info(f"Recoding video track for: {x}")
            video[0] = change_bitrate(video[0], local_params.recode)
        output_file = str(Path(x).with_suffix('.mkv'))
        merge_tracks(video, audio, subtitle, output_file)
        logging.info(f"Successfully processed: {x}")
    except Exception as e:
        logging.error(f"Failed to process {x}: {e}")
        raise
    finally:
        delete_tracks(video, audio, subtitle)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py <directory> [bitrate]")
        sys.exit(1)

    dir_to_process = Path(sys.argv[1])
    dir1 = dir_to_process.is_dir()
    params = ProcessingParams(
        recode=f'{sys.argv[2]}K' if len(sys.argv) > 2 else None
    )

    try:
        files = []
        if dir1:
            for f in dir_to_process.rglob("*"):
                if f.is_file() and is_video_file(f):
                    files.append(str(f))
        else:
            files.append(sys.argv[1])

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
