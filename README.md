# 동영상 자막 생성 with OpenAI API

OpenAI transcriptions API를 이용하여 비디오(각종 .mov 파일)로부터 자막(.srt) 파일을 생성하는 프로그램입니다.

FFmpeg를 이용해 비디오에서 오디오를 내부로 추출하고, OpenAI transcriptions API를 이용해 자막을 내용으로 생성합니다.

---
## Dependencies
이 프로젝트는 다음 환경에서 실험되었습니다:
```
Python 3.9 이상  
macOS 환경
```

---
## 사용법

### OpenAI API 키 설정

.env 파일에 OpenAI API 키를 입력하세요:

```env
OPENAI_API_KEY="sk-proj-..."
```

### 가상환경 설치

```bash
pip install -r requirements.txt
```
혹은
```bash
uv add -r requirements.txt
```


### 실행

```bash
python main.py --video /path/to/video.mov
python main.py -v /path/to/video.mov
```
혹은
```bash
uv run main.py --video /path/to/video.mov
uv run main.py -v /path/to/video.mov
```


---

## 키 옵션

| 옵션 | 값(기본값) | 설명 |
|--------|--------------------|--------|
| `--video, -v` | *(필수)* | 변환할 비디오 파일의 경로 |
| `--output, -o` | same as video name | 생성할 SRT 파일의 경로 |
| `--model, -m` | `whisper-1` | 사용할 OpenAI transcript 모델 |
| `--language, -l` | `ko` | 음성 인식 언어 |
| `--chunk-duration` | `10 * 60 * 1000` | 청크 지속시간 (ms) |

---
## 기능 요약
- **ffmpeg**로 비디오에서 오디오 추출
- **asyncio + AsyncOpenAI**로 OpenAI transcriptions API 비동기 호출
- .srt 자막 파일 자동 생성

---

## 주의사항
- OpenAI transcriptions API는 파일 당 25MB 제한이 있습니다. 본 스크립트는 자동으로 10분씩 오디오를 분할하여 대응합니다.
- 긴 영상의 경우 처리 시간이 오래 걸릴 수 있습니다 (API 호출 수 증가).

---
## 라이선스

MIT License
