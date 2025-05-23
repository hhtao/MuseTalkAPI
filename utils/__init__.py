# Make utils directory a package and export main functions

# Audio utilities
from utils.audio_utils import (
    gpt_sovits_tts,
    fallback_tts,
    get_audio_duration,
    create_bytes_stream
)

# Subtitle utilities
from utils.subtitle_utils import (
    calculate_mixed_subtitle_timing,
    generate_srt_file,
    generate_vtt_file,
    add_subtitles_to_video,
    split_text_to_sentences,
    validate_text_cut_method,
    should_use_mixed_method,
    create_subtitle_for_text,
    get_video_duration
)

# GPU management
from utils.gpu_manager import (
    initialize_multi_gpu_system,
    assign_task_to_gpu,
    process_batch_slides,
    get_gpu_status,
    clean_gpu_cache
)

# Task management
from utils.task_manager import (
    task_progress,
    create_task,
    update_task_progress,
    get_task_status,
    get_task_result,
    initialize_task_manager,
    get_active_tasks,
    get_all_tasks_status,
    mark_task_error
)

# Video generation
from utils.video_generator import (
    get_random_avatar,
    process_batch_slide_task
)
