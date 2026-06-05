from __future__ import annotations

from matrix_fsdp.core.layout import LayoutSegment


def coalesce_contiguous_segments(segments: tuple[LayoutSegment, ...]) -> tuple[LayoutSegment, ...]:
    if not segments:
        return ()

    fused: list[LayoutSegment] = []
    current = segments[0]
    for segment in segments[1:]:
        if current.global_end == segment.global_start and current.local_end == segment.local_start:
            current = LayoutSegment(
                global_start=current.global_start,
                global_end=segment.global_end,
                local_start=current.local_start,
            )
            continue
        fused.append(current)
        current = segment
    fused.append(current)
    return tuple(fused)


def coalesce_rank_segments(
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
) -> tuple[tuple[LayoutSegment, ...], ...]:
    return tuple(coalesce_contiguous_segments(segments) for segments in rank_segments)
