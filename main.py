#!/usr/bin/env python3
import os
import sys
import argparse
import asyncio
import datetime
import logging
import tempfile
from typing import List

import ffmpeg
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.audio import TranscriptionVerbose
from pydub import AudioSegment


logger = logging.getLogger(__name__)
load_dotenv()


def extract_audio(video_path: str, audio_path: str) -> None:
    """
    비디오 파일에서 오디오를 추출합니다.

    Args:
        video_path (str): 입력 비디오 파일 경로.
        audio_path (str): 출력 오디오 파일 경로.
    """
    logger.info(f"오디오 추출 시작: {video_path}")
    try:
        ffmpeg.input(video_path).output(
            audio_path,
            acodec="pcm_s16le",  # 16-bit PCM 오디오 코덱
            ac=1,  # 모노 오디오 (1채널)
            ar="16k",  # 16kHz 샘플링 레이트
        ).overwrite_output().run()
        logger.info(f"오디오 추출 완료: {audio_path}")
    except Exception as e:
        logger.error(f"오디오 추출 실패: {str(e)}")
        raise


def split_audio_by_duration(audio_path: str, chunk_duration: int) -> List[AudioSegment]:
    """
    오디오 파일을 일정 시간 간격으로 분할합니다.

    Args:
        audio_path (str): 입력 오디오 파일 (.wav)
        chunk_duration (int): 청크 길이 (밀리초 단위)

    Returns:
        List[AudioSegment]: 분할된 오디오 청크 리스트
    """
    logger.info(f"오디오 분할 시작 (시간 기준): {audio_path}")

    try:
        audio = AudioSegment.from_wav(audio_path)
        total_length = len(audio)

        chunks = []
        for start in range(0, total_length, chunk_duration):
            end = min(start + chunk_duration, total_length)
            chunks.append(audio[start:end])

        logger.info(f"오디오 분할 완료: {len(chunks)}개 청크")
        return chunks

    except Exception as e:
        logger.error(f"오디오 시간 분할 실패: {str(e)}")
        raise


async def process_chunk(
    client: AsyncOpenAI, chunk_path: str, model: str, language: str
) -> TranscriptionVerbose:
    """
    단일 오디오 청크를 비동기적으로 처리합니다.

    Args:
        client (AsyncOpenAI): OpenAI 비동기 클라이언트.
        chunk_path (str): 오디오 청크 파일 경로.
        model (str): 사용할 Whisper 모델.
        language (str): 인식할 언어.

    Returns:
        TranscriptionVerbose: 전사 결과.
    """
    with open(chunk_path, "rb") as audio_file:
        transcription = await client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="verbose_json",
            language=language,
        )
        return transcription


async def process_audio_chunks(
    chunks: List[AudioSegment], client: AsyncOpenAI, model: str, language: str
) -> List[TranscriptionVerbose]:
    """
    분할된 오디오 청크들을 Whisper API로 비동기 처리합니다.

    Args:
        chunks (List[AudioSegment]): 분할된 오디오 청크 리스트.
        client (AsyncOpenAI): OpenAI 비동기 클라이언트.
        model (str): 사용할 Whisper 모델.
        language (str): 인식할 언어.

    Returns:
        List[TranscriptionVerbose]: 각 청크의 전사 결과 리스트.
    """
    logger.info(f"오디오 처리 시작: {len(chunks)}개 청크")

    results = {}
    # TemporaryDirectory를 사용하여 임시 파일 관리
    with tempfile.TemporaryDirectory() as temp_dir:
        chunk_paths = []
        # 모든 청크를 임시 파일로 저장
        for i, chunk in enumerate(chunks):
            chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
            chunk.export(chunk_path, format="wav")
            chunk_paths.append((i, chunk_path))

        # 각 청크에 대한 비동기 태스크 생성
        tasks = [
            asyncio.create_task(
                process_chunk(client, path, model, language), name=f"chunk_{i}"
            )
            for i, path in chunk_paths
        ]

        total_tasks = len(tasks)
        completed_tasks = 0

        # 진행률 모니터링하며 결과 수집
        while tasks:
            done, pending = await asyncio.wait(
                tasks, timeout=5, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                try:
                    idx = int(task.get_name().split("_")[1])
                    results[idx] = task.result()
                    completed_tasks += 1
                    logger.info(f"처리 진행: {completed_tasks}/{total_tasks}")
                except Exception as e:
                    logger.error(f"태스크 결과 처리 중 오류: {str(e)}")
            tasks = list(pending)

    # 결과를 인덱스 순서대로 정렬하여 반환
    transcriptions = [results[i] for i in sorted(results.keys())]
    logger.info(f"오디오 처리 완료: {len(transcriptions)}개 결과")
    return transcriptions


def merge_transcriptions(
    transcriptions: List[TranscriptionVerbose], chunk_duration: int
) -> dict:
    """
    여러 전사 결과를 하나로 병합합니다.

    각 청크의 시작 시간을 chunk_duration 만큼 오프셋하여 세그먼트를 정렬합니다.

    Args:
        transcriptions (List[TranscriptionVerbose]): 전사 결과 리스트.
        chunk_duration (int): 각 청크의 길이 (밀리초 단위).

    Returns:
        dict: 병합된 전사 결과 (세그먼트 리스트 포함).
    """
    logger.info(f"전사 결과 병합 시작: {len(transcriptions)}개")
    merged = {"segments": []}

    try:
        # 각 청크의 세그먼트를 시작 시간에 맞춰 병합
        for i, transcription in enumerate(transcriptions):
            # 각 청크의 시작 시간 계산
            chunk_start = chunk_duration / 1000 * i

            for segment in transcription.segments:
                # 시간 오프셋 조정
                adjusted_segment = {
                    "start": segment.start + chunk_start,
                    "end": segment.end + chunk_start,
                    "text": segment.text,
                }
                merged["segments"].append(adjusted_segment)

        logger.info(f"전사 결과 병합 완료: {len(merged['segments'])}개 세그먼트")
        return merged
    except Exception as e:
        logger.error(f"전사 결과 병합 실패: {str(e)}")
        raise


def format_timestamp(seconds: float) -> str:
    """
    초 단위의 시간을 SRT 포맷(00:00:00,000)의 문자열로 변환합니다.

    Args:
        seconds (float): 변환할 초 단위 시간

    Returns:
        str: "HH:MM:SS,mmm" 형식의 타임스탬프 문자열
    """
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def generate_srt(transcription: dict, output_file_path: str) -> None:
    """
    Whisper API 응답을 SRT 자막 파일로 생성합니다.

    Args:
        transcription (dict): 병합된 전사 결과.
        output_file_path (str): 생성할 SRT 파일 경로.
    """
    logger.info(f"SRT 파일 생성 시작: {output_file_path}")
    segments = transcription["segments"]

    with open(output_file_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")
    logger.info(f"SRT 파일 생성 완료: {output_file_path}")


async def async_main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="비디오 파일에서 자막을 생성합니다.")
    parser.add_argument("--video", "-v", required=True, help="변환할 비디오 파일 경로")
    parser.add_argument("--output", "-o", help="생성할 SRT 파일의 경로")
    parser.add_argument(
        "--model", "-m", default="whisper-1", help="사용할 transcriptions 모델"
    )
    parser.add_argument("--language", "-l", default="ko", help="음성 인식 언어")
    parser.add_argument(
        "--chunk-duration", type=int, default=10 * 60 * 1000, help="청크 지속시간 (ms)"
    )
    args = parser.parse_args()

    # 파일 경로 설정
    video_path = args.video
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    srt_path = args.output if args.output else os.path.splitext(video_path)[0] + ".srt"

    logger.info("=" * 80)
    logger.info("OpenAI API 자막 생성 시작")
    logger.info(f"비디오 파일: {video_path}")
    logger.info(f"임시 오디오 파일: {audio_path}")
    logger.info(f"출력 SRT 파일: {srt_path}")
    logger.info(f"사용 모델: {args.model}")
    logger.info(f"음성 인식 언어: {args.language}")
    logger.info(f"청크 지속시간: {args.chunk_duration}ms")
    logger.info("=" * 80)

    try:
        # 오디오 추출
        extract_audio(video_path, audio_path)

        # 오디오 분할
        chunks = split_audio_by_duration(audio_path, args.chunk_duration)

        # OpenAI 비동기 클라이언트 초기화
        client = AsyncOpenAI()

        # 각 청크 처리
        transcriptions = await process_audio_chunks(
            chunks, client, args.model, args.language
        )

        # 전사 결과 병합
        merged_transcription = merge_transcriptions(transcriptions, args.chunk_duration)

        # SRT 파일 생성
        generate_srt(merged_transcription, srt_path)

        logger.info(f"작업 완료: {srt_path}")

    except Exception as e:
        logger.error(f"작업 실패: {str(e)}")
        sys.exit(1)

    finally:
        # 임시 오디오 파일 삭제
        if os.path.exists(audio_path):
            os.remove(audio_path)

        logger.info("작업 종료")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    # 로그 디렉토리 생성
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 로깅 설정
    log_filename = (
        f"openai_transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    main()
